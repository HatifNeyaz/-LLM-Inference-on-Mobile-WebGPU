const logEl = document.getElementById("log");

function log(msg) {
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
    console.log(msg);
}

// ==========================================
// KERNEL A: Naive GEMM
// ==========================================
const naiveWGSL = `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Uniforms { M: u32, N: u32, K: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= uniforms.M || col >= uniforms.N) { return; }

    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < uniforms.K; k = k + 1u) {
        sum = sum + A[row * uniforms.K + k] * B[k * uniforms.N + col];
    }
    C[row * uniforms.N + col] = sum;
}
`;

// ==========================================
// KERNEL B: Tiled Shared Memory GEMM
// ==========================================
const tiledWGSL = `
const TILE_SIZE = 16u;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Uniforms { M: u32, N: u32, K: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_y = local_id.y;
    let local_x = local_id.x;

    var sum: f32 = 0.0;
    let numTiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_x;
        if (row < uniforms.M && a_col < uniforms.K) {
            tileA[local_y * TILE_SIZE + local_x] = A[row * uniforms.K + a_col];
        } else {
            tileA[local_y * TILE_SIZE + local_x] = 0.0;
        }

        let b_row = t * TILE_SIZE + local_y;
        if (col < uniforms.N && b_row < uniforms.K) {
            tileB[local_y * TILE_SIZE + local_x] = B[b_row * uniforms.N + col];
        } else {
            tileB[local_y * TILE_SIZE + local_x] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[local_y * TILE_SIZE + k] * tileB[k * TILE_SIZE + local_x];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        C[row * uniforms.N + col] = sum;
    }
}
`;

// ==========================================
// KERNEL C: Vectorized vec4 GEMM
// ==========================================
const vectorizedWGSL = `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec4<f32>>;

struct Uniforms { M: u32, N: u32, K: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col_vec = global_id.x;

    let N_vecs = uniforms.N / 4u;
    if (row >= uniforms.M || col_vec >= N_vecs) { return; }

    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < uniforms.K; k = k + 1u) {
        let a_val = A[row * uniforms.K + k];
        let b_vec = B[k * N_vecs + col_vec];
        sum = sum + a_val * b_vec;
    }
    C[row * N_vecs + col_vec] = sum;
}
`;

// ==========================================
// KERNEL D: Fused Q4 Dequantization + GEMM
// ==========================================
const fusedDequantWGSL = `
// B contains packed 4-bit weights: 8 weights per u32
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Uniforms { M: u32, N: u32, K: u32 };
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const SCALE: f32 = 0.05; // Fixed affine dequantization scale
const ZERO_POINT: f32 = 8.0;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= uniforms.M || col >= uniforms.N) { return; }

    var sum: f32 = 0.0;
    let K_packed = uniforms.K / 8u;

    for (var kp = 0u; kp < K_packed; kp = kp + 1u) {
        let packed_val = B_packed[kp * uniforms.N + col];
        let base_k = kp * 8u;

        // Unpack eight 4-bit signed/unsigned weights and multiply-accumulate on the fly
        for (var i = 0u; i < 8u; i = i + 1u) {
            let shift = i * 4u;
            let raw_nibble = f32((packed_val >> shift) & 0x0Fu);
            let weight = (raw_nibble - ZERO_POINT) * SCALE;
            sum = sum + A[row * uniforms.K + (base_k + i)] * weight;
        }
    }
    C[row * uniforms.N + col] = sum;
}
`;

function calculateCPU(M, N, K, A, B) {
    const C = new Float32Array(M * N);
    for (let r = 0; r < M; r++) {
        for (let c = 0; c < N; c++) {
            let sum = 0;
            for (let k = 0; k < K; k++) {
                sum += A[r * K + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
    return C;
}

function verifyCorrectness(M, N, resultC, expectedC) {
    let maxError = 0;
    const samples = Math.min(1000, M * N);
    for (let i = 0; i < samples; i++) {
        const idx = Math.floor(Math.random() * (M * N));
        const diff = Math.abs(resultC[idx] - expectedC[idx]);
        if (diff > maxError) maxError = diff;
    }
    return maxError;
}

async function runBenchmark(M) {
    const kernelType = document.getElementById("sel-kernel").value;
    const K = 2048;
    const N = 2048;

    log(`\n=================================================`);
    log(`Executing Kernel ${kernelType} | Shape: M=${M}, N=${N}, K=${K}`);
    log(`=================================================`);

    const adapter = await navigator.gpu.requestAdapter();
    const hasTimestamp = adapter.features.has('timestamp-query');
    const hasShaderF16 = adapter.features.has('shader-f16');
    log(`[Features] timestamp-query: ${hasTimestamp} | shader-f16: ${hasShaderF16}`);

    const requiredFeatures = [];
    if (hasTimestamp) requiredFeatures.push('timestamp-query');

    const device = await adapter.requestDevice({ requiredFeatures });

    // Generate Inputs
    const A = new Float32Array(M * K).map(() => Math.random() * 2 - 1);
    let B_raw = new Float32Array(K * N);
    let B_packed = null;

    if (kernelType === 'D') {
        // Generate simulated Q4 weights packed 8 per 32-bit uint
        const SCALE = 0.05;
        const ZERO_POINT = 8.0;
        const numPacked = (K / 8) * N;
        B_packed = new Uint32Array(numPacked);

        for (let kp = 0; kp < K / 8; kp++) {
            for (let c = 0; c < N; c++) {
                let pack = 0;
                for (let i = 0; i < 8; i++) {
                    const rawVal = Math.floor(Math.random() * 16);
                    pack |= (rawVal & 0x0F) << (i * 4);
                    // Match reference weights for CPU validation
                    B_raw[(kp * 8 + i) * N + c] = (rawVal - ZERO_POINT) * SCALE;
                }
                B_packed[kp * N + c] = pack;
            }
        }
    } else {
        B_raw = B_raw.map(() => Math.random() * 2 - 1);
    }

    const sizeA = A.byteLength;
    const sizeB = (kernelType === 'D') ? B_packed.byteLength : B_raw.byteLength;
    const sizeC = M * N * 4;

    const bufferA = device.createBuffer({ size: sizeA, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufferB = device.createBuffer({ size: sizeB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufferC = device.createBuffer({ size: sizeC, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(bufferA, 0, A);
    device.queue.writeBuffer(bufferB, 0, (kernelType === 'D') ? B_packed : B_raw);
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([M, N, K]));

    let wgsl;
    if (kernelType === 'A') wgsl = naiveWGSL;
    else if (kernelType === 'B') wgsl = tiledWGSL;
    else if (kernelType === 'C') wgsl = vectorizedWGSL;
    else if (kernelType === 'D') wgsl = fusedDequantWGSL;

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: 'main' }
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferC } },
            { binding: 3, resource: { buffer: uniformBuffer } }
        ]
    });

    const querySet = hasTimestamp ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const resolveBuffer = hasTimestamp ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
    const resultBuffer = hasTimestamp ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;

    // Warm-up iteration to compile pipeline state
    {
        const warmupEncoder = device.createCommandEncoder();
        const warmupPass = warmupEncoder.beginComputePass();
        warmupPass.setPipeline(pipeline);
        warmupPass.setBindGroup(0, bindGroup);
        const divX = (kernelType === 'C') ? 4 : 1;
        warmupPass.dispatchWorkgroups(Math.ceil((N / divX) / 16), Math.ceil(M / 16));
        warmupPass.end();
        device.queue.submit([warmupEncoder.finish()]);
    }

    // Benchmark Pass with Timestamps
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass({
        timestampWrites: hasTimestamp ? { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined
    });
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const dispatchDivisorX = (kernelType === 'C') ? 4 : 1;
    passEncoder.dispatchWorkgroups(Math.ceil((N / dispatchDivisorX) / 16), Math.ceil(M / 16));
    passEncoder.end();

    if (hasTimestamp) {
        commandEncoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(resolveBuffer, 0, resultBuffer, 0, 16);
    }

    const readbackBuffer = device.createBuffer({ size: sizeC, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    commandEncoder.copyBufferToBuffer(bufferC, 0, readbackBuffer, 0, sizeC);

    device.queue.submit([commandEncoder.finish()]);

    if (hasTimestamp) {
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const times = new BigInt64Array(resultBuffer.getMappedRange());
        const durationNs = Number(times[1] - times[0]);
        resultBuffer.unmap();

        const durationMs = durationNs / 1e6;
        const durationSec = durationNs / 1e9;
        const gflops = ((2 * M * N * K) / durationSec) / 1e9;
        const gbPerSec = ((sizeA + sizeB + sizeC) / durationSec) / 1e9;

        log(`Time        : ${durationMs.toFixed(3)} ms`);
        log(`Performance : ${gflops.toFixed(2)} GFLOPS`);
        log(`Memory B/W  : ${gbPerSec.toFixed(2)} GB/s`);

        log("Validating correctness against CPU...");
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const gpuResult = new Float32Array(readbackBuffer.getMappedRange());
        const cpuExpected = calculateCPU(M, N, K, A, B_raw);
        const err = verifyCorrectness(M, N, gpuResult, cpuExpected);
        readbackBuffer.unmap();

        log(`Max Abs Err : ${err.toExponential(4)}`);
    }
}

document.getElementById("btn-decode").addEventListener("click", () => runBenchmark(1));
document.getElementById("btn-prefill").addEventListener("click", () => runBenchmark(128));