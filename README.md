# WebGPU LLM Profiling and Custom GEMM Kernels for Gemma 2B

**Author:** Hatif Neyaz  
**Project:** LLM Inference on Mobile WebGPU (Part 1: Profiling & Part 2: Custom GEMM Kernels)

---

## Table of Contents
1. [Overview & Execution Environment](#overview--execution-environment)
2. [Part 1: Baseline Profiling & Bottleneck Analysis](#part-1-baseline-profiling--bottleneck-analysis)
   - [Target Model Variant](#target-model-variant)
   - [Tokenizer Overhead Note](#tokenizer-overhead-note)
   - [Baseline Benchmark Results](#baseline-benchmark-results)
   - [Command Stream & Kernel Breakdown](#command-stream--kernel-breakdown)
   - [Bottleneck Classification](#bottleneck-classification)
   - [Profiling Constraints & Missing Metrics](#profiling-constraints--missing-metrics)
3. [Part 2: Custom WGSL GEMM Kernels](#part-2-custom-wgsl-gemm-kernels)
   - [Matrix Multiplication Primer (M, N, K)](#matrix-multiplication-primer-m-n-k)
   - [Kernel Implementations (A through D)](#kernel-implementations-a-through-d)
   - [GEMM Benchmark Results](#gemm-benchmark-results)
   - [Detailed Comparative Analysis](#detailed-comparative-analysis)
   - [Numerical Verification & Correctness](#numerical-verification--correctness)
4. [Step-by-Step Reproducibility Guide](#step-by-step-reproducibility-guide)
   - [Prerequisites](#prerequisites)
   - [Directory Structure](#directory-structure)
   - [Step 1: Start the Local Server](#step-1-start-the-local-server)
   - [Step 2: Run Part 1 (WebLLM Profiling)](#step-2-run-part-1-webllm-profiling)
   - [Step 3: Run Part 2 (Custom GEMM Benchmark)](#step-3-run-part-2-custom-gemm-benchmark)

---

## Overview & Execution Environment

This project investigates browser-based edge LLM inference using WebGPU. It profiles an existing WebLLM stack on Gemma 2B (Part 1) and implements custom Matrix Multiplication (GEMM) compute shaders in WebGPU Shading Language (WGSL) to address memory-bandwidth and dequantization bottlenecks (Part 2).

* **Target Hardware Deviation:** The assignment specified Snapdragon devices with Qualcomm Adreno GPUs. Due to physical hardware access limits and restricted device cloud availability, testing was conducted on a local workstation:
  - **CPU:** AMD Ryzen 5 4600H (6 cores, 12 threads)
  - **GPU:** NVIDIA GeForce GTX 1660 Ti (6 GB GDDR6 VRAM)
  - **System RAM:** 8 GB
  - **OS / Environment:** Windows Subsystem for Linux (WSL2 / Ubuntu)
  - **Browser:** Chromium-based browser with WebGPU enabled
* **WebGPU Feature Verification:**
  - `timestamp-query`: **Supported (true)** (Nanosecond-level GPU-side hardware timers)
  - `shader-f16`: **Supported (true)** (Native 16-bit half-precision floating-point arithmetic)

---

## Part 1: Baseline Profiling & Bottleneck Analysis

### Target Model Variant
The assigned target model—Gemma E2B (Gemma 3n)—is not packaged in official WebLLM precompiled binaries. The closest production variant, **`gemma-2-2b-it-q4f16_1-MLC`**, was evaluated. It stores weights in 4-bit affine quantization (`q4`) and executes activations in 16-bit float (`f16`) across 26 Transformer layers.

### Tokenizer Overhead Note
Fixed prompt targets of 128, 512, and 1024 produced actual token counts of **138, 522, and 1034** respectively. This systematic 10-token overhead is caused by the Gemma tokenizer prepending and appending mandatory control tokens (`<bos>`, `<start_of_turn>`, `user`, `\n`, `<end_of_turn>`, `<start_of_turn>model\n`).

### Baseline Benchmark Results
Each context length was evaluated across 5 consecutive runs with automated pipeline warmup:

| Metric | Target 128 (Actual: 138) | Target 512 (Actual: 522) | Target 1024 (Actual: 1034) |
| :--- | :--- | :--- | :--- |
| **Prefill Latency (Median)** | **1.206 s** | **4.524 s** | **11.327 s** |
| **Prefill Latency (p95)** | 1.440 s | 4.808 s | 11.548 s |
| **Decode Throughput (Median)**| **5.81 tok/s** | **5.93 tok/s** | **5.96 tok/s** |
| **Decode Throughput (p95)** | 6.63 tok/s | 6.25 tok/s | 6.20 tok/s |
| **Per-Token Decode Latency (Median)** | 158.41 ms | 166.83 ms | 159.45 ms |
| **Per-Token Decode Latency (Range)** | 150.17 ms – 174.45 ms | 158.49 ms – 174.40 ms | 158.40 ms – 166.47 ms |
| **Total Dispatches (20 tokens)** | 6,821 dispatches | 6,821 dispatches | 7,190 dispatches |
| **Dispatches per Decode Token** | **~341 dispatches/token** | **~341 dispatches/token** | **~359 dispatches/token** |
| **V8 JS Heap Memory** | ~600 MB | ~589 MB | ~580 MB |

### Command Stream & Kernel Breakdown
By intercepting `GPUDevice.prototype.createComputePipeline` and `GPUComputePassEncoder.prototype.dispatchWorkgroups`, every kernel pass was classified by operational category:

| Kernel Category | Representative Kernels | Count (1034 Tokens) | Architectural Purpose |
| :--- | :--- | :--- | :--- |
| **Compute (GEMM)** | `fused_dequantize_NT_matmul`, `fused_NT_matmul` | 3,009 | Projection layers ($Q, K, V, O$, Gate, Up, Down) |
| **RMSNorm** | `fuse_add_norm_prefill`, `rms_norm1`, `rms_norm2` | 2,225 | Pre-attention, post-attention, and final layer norms |
| **KV Cache & Attention** | `batch_decode_paged_kv`, `batch_prefill_ragged_kv` | 1,144 | Paged KV cache appending and multi-head attention |
| **RoPE (Positional)** | `fused_rope_kernel` | 546 | Rotary positional embedding passes |
| **Other / Utilities** | `fused_split_gelu_tanh`, `chunk_lse`, `argsort` | 266 | Activation functions, softmax reduction, top-p/top-k |

*Decode Dispatch Signature:* During generation of a single token, exactly 26 layers $\times$ 4 projections = **104 GEMM dispatches** occur alongside 53 RMSNorm dispatches and 26 Paged Attention dispatches.

### Bottleneck Classification
1. **Dominant Bottleneck: Memory Bandwidth (Decode Phase)**
   - *Evidence:* Decode throughput stays flat at ~5.8 to 5.9 tok/s across all prompt sizes. At $M=1$, every generated token must stream the entire ~1.3 GB weight set over the bus. The achieved bandwidth of 7.8–8.5 GB/s is bounded by memory transaction latency for single-vector operations ($FLOPs/byte \approx 1$).
2. **Secondary Bottleneck: Kernel Launch Overhead**
   - *Evidence:* Over 340 separate dispatches are submitted for every single token. The CPU-side WebGPU validation and Dawn serialization overhead creates execution bubbles between rapid, lightweight kernels like RMSNorm and RoPE.
3. **Prefill Scaling: Compute Bound**
   - *Evidence:* Prefill latency scales linearly with context length (1.2 s $\to$ 4.5 s $\to$ 11.3 s). With $M \ge 128$, weight matrices are amortized across multiple rows, saturating GPU compute units and yielding ~91 tok/s during prompt processing.

### Profiling Constraints & Missing Metrics
* **Memory Usage (Metric 4):** Standard Web APIs isolate underlying hardware allocations. The V8 engine exposed ~600 MB of JS heap, but GPU VRAM buffers and process RSS could not be captured programmatically without native platform diagnostics.
* **Individual Kernel Duration (Metric 6):** Microsecond-level individual kernel execution times via `chrome://tracing` could not be separated because Chromium's Dawn pipeline bundled dispatches under serialized IPC calls without distinct labels.

---

## Part 2: Custom WGSL GEMM Kernels

### Matrix Multiplication Primer ($M, N, K$)
Matrix multiplication $C = A \times B$ underpins all Transformer projection layers:
* **$M$ (Batch / Token Count):** Number of tokens processed simultaneously. $M = 1$ in single-token Decode. $M = 128$ in batched prompt Prefill.
* **$K$ (Input Dimension):** Gemma 2B hidden dimension width ($K = 2048$).
* **$N$ (Output Dimension):** Layer projection dimension width ($N = 2048$).

### Kernel Implementations (A through D)
* **Kernel A (Naive Baseline):** Basic scalar WGSL dot product where each GPU thread computes one element of $C$ by pulling values directly from global storage buffers.
* **Kernel B (Tiled GEMM):** Workgroups load $16 \times 16$ tiles into on-chip workgroup shared memory (`var<workgroup>`), synchronizing via `workgroupBarrier()` to reuse values across threads.
* **Kernel C (Vectorized Loads):** Uses `vec4<f32>` memory types to load 128 bits (4 floats) per memory transaction, aligning with hardware memory bus widths.
* **Kernel D (Fused Q4 Dequant + GEMM):** Weights are stored as packed 4-bit integers (8 weights per 32-bit `u32`). The shader performs bit-shifting (`>>`), masking (`& 0x0F`), and scale adjustments directly in registers on-the-fly, eliminating separate dequantization passes.

### GEMM Benchmark Results
Tested on Gemma 2B layer shapes ($N=2048, K=2048$) using WebGPU `timestamp-query`:

| Kernel | Description | Shape $(M, N, K)$ | Workgroup | Time (ms) | GFLOPS | Bandwidth (GB/s) | Max Absolute Error vs FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Naive Baseline | $(1, 2048, 2048)$ | $16 \times 16$ | 10.608 | 0.79 | 1.58 | $8.39 \times 10^{-5}$ |
| **A** | Naive Baseline | $(128, 2048, 2048)$ | $16 \times 16$ | 53.641 | 20.02 | 0.35 | $7.63 \times 10^{-5}$ |
| **B** | Shared Memory Tiling | $(1, 2048, 2048)$ | $16 \times 16$ | 2.886 | 2.91 | 5.82 | $7.25 \times 10^{-5}$ |
| **B** | Shared Memory Tiling | $(128, 2048, 2048)$ | $16 \times 16$ | 20.237 | **53.06** | 0.93 | $9.16 \times 10^{-5}$ |
| **C** | Vectorized (`vec4`) | $(1, 2048, 2048)$ | $16 \times 16$ | 5.900 | 1.42 | 2.85 | $6.10 \times 10^{-5}$ |
| **C** | Vectorized (`vec4`) | $(128, 2048, 2048)$ | $16 \times 16$ | 24.632 | 43.59 | 0.77 | $8.77 \times 10^{-5}$ |
| **D** | Fused Q4 Dequant | $(1, 2048, 2048)$ | $16 \times 16$ | **1.581** | **5.31** | 1.34 | $2.67 \times 10^{-5}$ |
| **D** | Fused Q4 Dequant | $(128, 2048, 2048)$ | $16 \times 16$ | 21.118 | 50.84 | 0.20 | $2.48 \times 10^{-5}$ |

### Detailed Comparative Analysis
* **Why Kernel D Dominates Decode ($M=1$):** At $M=1$, the operation is severely memory bandwidth bound. Storing Matrix B as 4-bit integers compresses its memory footprint by $8\times$. Unpacking integer nibbles on-the-fly inside ALU registers avoids memory stalls, cutting latency by **85%** (from 10.608 ms down to 1.581 ms).
* **Why Kernel B Dominates Prefill ($M=128$):** At $M=128$, the workload shifts to compute bound. Staging data in workgroup shared memory allows 256 threads to share weights without revisiting global VRAM, achieving the highest throughput at **53.06 GFLOPS**.
* **Why Kernel C Trails Kernel B:** While `vec4` loads maximize bus bandwidth, Kernel C did not stage data into shared memory. Threads still stalled on global memory access compared to Kernel B's on-chip SRAM cache.

### Numerical Verification & Correctness
Every kernel run was compared against an FP32 matrix multiplication reference computed on the CPU. The maximum absolute error across all kernels and shapes remained between **$2.48 \times 10^{-5}$ and $9.16 \times 10^{-5}$**, well within single-precision floating-point tolerances.

---

## Step-by-Step Reproducibility Guide

### Prerequisites
* **Node.js** (v18.0.0 or higher)
* **Chromium Browser** (Google Chrome or Microsoft Edge) with WebGPU enabled
* **GPU Hardware** with up-to-date graphics drivers

### Directory Structure
```text
mobile-llm-profiling/
├── index.html        # Part 1 UI
├── main.js           # Part 1 WebLLM harness & interceptor
├── part2.html        # Part 2 Custom GEMM benchmark UI
├── part2.js          # Part 2 Custom WGSL kernels (A–D) & validation
├── server.js         # HTTP server with COOP/COEP headers
└── README.md
```

### Step 1: Start the Local Server
WebAssembly threads and WebGPU timestamp queries require Cross-Origin Isolation headers (`COOP: same-origin`, `COEP: require-corp`). Start the local server from your project root:

```bash
cd mobile-llm-profiling
node server.js
```

The terminal will confirm:
```text
Profiling Server running at http://localhost:8080
Make sure to open this in Chrome/Edge.
```

### Step 2: Run Part 1 (WebLLM Profiling)
1. Open Chrome or Edge and navigate to:
   ```text
   http://localhost:8080
   ```
2. Open DevTools (**F12**) and switch to the **Console** tab.
3. Wait for weight downloading and shader compilation. The status banner will update to:
   ```text
   Status: Engine Ready!
   ```
4. Click **Run 128 (5x Sweep)**:
   - On-screen log displays Prefill Latency, Decode Throughput, and Token Latency.
   - DevTools console outputs the complete kernel breakdown table with classified operational categories.
5. Click **Run 512 (5x Sweep)** and **Run 1024 (5x Sweep)** to complete all sweep context lengths.
6. Click **Export Results JSON** to save structured profiling data locally.

### Step 3: Run Part 2 (Custom GEMM Benchmark)
1. In the same browser, open a new tab and navigate to:
   ```text
   http://localhost:8080/part2.html
   ```
2. Select the desired kernel from the dropdown:
   - **Kernel A (Naive WGSL)**
   - **Kernel B (Tiled 16x16 Shared Memory)**
   - **Kernel C (Vectorized vec4 Loads)**
   - **Kernel D (Fused Q4 Dequant + GEMM)**
3. Click **Run Decode (M=1)** to test single-token projection throughput ($1 \times 2048 \times 2048$).
4. Click **Run Prefill (M=128)** to test batched prompt projection throughput ($128 \times 2048 \times 2048$).
5. Inspect the benchmark log for GPU execution time (ms), GFLOPS, memory bandwidth, and max absolute error vs. CPU reference.