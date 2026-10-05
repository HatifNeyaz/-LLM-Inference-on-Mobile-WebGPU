// import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

// const logEl = document.getElementById("metrics-log");
// const statusEl = document.getElementById("status");
// let engine;

// function logMessage(msg) {
//     logEl.textContent += msg + "\n";
//     console.log(msg);
// }

// async function initializeWebLLM() {
//     statusEl.textContent = "Status: Initializing GPU & Downloading Model...";
    
//     try {
//         // Request timestamp-query for Metric 6
//         const adapter = await navigator.gpu?.requestAdapter();
//         if (adapter && !adapter.features.has('timestamp-query')) {
//             logMessage("WARNING: 'timestamp-query' not supported by this browser.");
//         }

//         // Using 1B model to avoid Windows TDR crashes during 128+ token prefills
//         engine = await CreateMLCEngine("gemma-2-2b-it-q4f16_1-MLC", {
//             initProgressCallback: (progress) => {
//                 statusEl.textContent = `Status: ${progress.text}`;
//             }
//         });

//         statusEl.textContent = "Status: Ready!";
//         document.getElementById("btn-128").disabled = false;
//         document.getElementById("btn-512").disabled = false;
//         document.getElementById("btn-1024").disabled = false;
//     } catch (error) {
//         statusEl.textContent = "Error initializing. Check console.";
//         console.error("Initialization error:", error);
//     }
// }

// async function runBenchmark(promptLength) {
//     logMessage(`\n--- Starting Benchmark: ${promptLength} tokens ---`);
//     const dummyPrompt = "word ".repeat(promptLength);
    
//     const tokenLatencies = [];
//     let lastTokenTime = null;
    
//     try {
//         const stream = await engine.chat.completions.create({
//             messages: [{ role: "user", content: dummyPrompt }],
//             temperature: 0.1,
//             max_tokens: 50,
//             stream: true,
//             stream_options: { include_usage: true }
//         });

//         for await (const chunk of stream) {
//             const now = performance.now();
            
//             if (lastTokenTime !== null) {
//                 tokenLatencies.push(now - lastTokenTime);
//             }
//             lastTokenTime = now;

//             if (chunk.usage) {
//                 const usage = chunk.usage;
//                 logMessage(`Prefill Latency: ${usage.extra.prefill_tokens_per_s ? (1/usage.extra.prefill_tokens_per_s * usage.prompt_tokens).toFixed(3) : 'N/A'} s`);
//                 logMessage(`Mean Decode Speed: ${usage.extra.decode_tokens_per_s.toFixed(2)} Tokens/sec`);
//             }
//         }

//         if (tokenLatencies.length > 0) {
//             tokenLatencies.sort((a, b) => a - b);
//             const median = tokenLatencies[Math.floor(tokenLatencies.length * 0.5)];
//             const p95 = tokenLatencies[Math.floor(tokenLatencies.length * 0.95)];
            
//             logMessage(`Decode Median Latency: ${median.toFixed(2)} ms/token`);
//             logMessage(`Decode P95 Latency: ${p95.toFixed(2)} ms/token`);
//         }
//     } catch (error) {
//         logMessage(`Error during benchmark: ${error.message}`);
//         console.error("Benchmark error:", error);
//     }
// }

// document.getElementById("btn-128").addEventListener("click", () => runBenchmark(128));
// document.getElementById("btn-512").addEventListener("click", () => runBenchmark(512));
// document.getElementById("btn-1024").addEventListener("click", () => runBenchmark(1024));

// initializeWebLLM();











import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm@0.2.82";

const logEl = document.getElementById("metrics-log");
const statusEl = document.getElementById("status");
let engine;

function logMessage(msg) {
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
    console.log(msg);
}

function calculatePercentiles(arr) {
    if (!arr || arr.length === 0) return { median: 0, p95: 0, min: 0, max: 0 };
    const sorted = [...arr].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length * 0.5)];
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    return {
        median,
        p95,
        min: sorted[0],
        max: sorted[sorted.length - 1]
    };
}

function getMemoryUsage() {
    const memory = {
        jsHeapUsedMB: null,
        jsHeapTotalMB: null
    };
    if (window.performance && performance.memory) {
        memory.jsHeapUsedMB = +(performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(2);
        memory.jsHeapTotalMB = +(performance.memory.totalJSHeapSize / (1024 * 1024)).toFixed(2);
    }
    return memory;
}

async function initializeWebLLM() {
    statusEl.textContent = "Status: Checking WebGPU adapter & initializing...";
    
    try {
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported on this browser.");
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No WebGPU adapter found.");
        }

        const hasTimestamp = adapter.features.has('timestamp-query');
        logMessage(`[System] Timestamp Query Supported: ${hasTimestamp}`);
        if (!hasTimestamp) {
            logMessage("[Notice] Browser lacks 'timestamp-query'. Per-kernel dispatches must be read via Chrome Tracing.");
        }

        // Target model configuration
        // Defaulting to gemma-2-2b-it-q4f16_1-MLC (or welcoma/gemma-4-E2B-it-q4f16_1-MLC)
        statusEl.textContent = "Status: Downloading weights & compiling pipeline...";
        engine = await CreateMLCEngine("gemma-2-2b-it-q4f16_1-MLC", {
            initProgressCallback: (progress) => {
                statusEl.textContent = `Status: ${progress.text}`;
            }
        });

        statusEl.textContent = "Status: Engine Ready!";
        document.getElementById("btn-128").disabled = false;
        document.getElementById("btn-512").disabled = false;
        document.getElementById("btn-1024").disabled = false;
        document.getElementById("btn-export").disabled = false;
    } catch (err) {
        statusEl.textContent = `Initialization failed: ${err.message}`;
        logMessage(`[Error] ${err.stack || err.message}`);
    }
}

// Stores structured results for final assignment reporting
window.profilingResults = {};

async function executeSingleRun(promptLength, runIndex, maxTokens = 30) {
    const dummyPrompt = "word ".repeat(promptLength);
    const tokenLatencies = [];
    let lastTokenTime = null;
    let prefillTimeSec = null;
    let decodeTokensPerSec = null;
    let totalPromptTokens = null;

    const memBefore = getMemoryUsage();
    const runStartTime = performance.now();

    const stream = await engine.chat.completions.create({
        messages: [{ role: "user", content: dummyPrompt }],
        temperature: 0.1,
        max_tokens: maxTokens,
        stream: true,
        stream_options: { include_usage: true }
    });

    for await (const chunk of stream) {
        const now = performance.now();
        if (lastTokenTime !== null) {
            tokenLatencies.push(now - lastTokenTime);
        }
        lastTokenTime = now;

        if (chunk.usage && chunk.usage.extra) {
            const usage = chunk.usage;
            totalPromptTokens = usage.prompt_tokens;
            if (usage.extra.prefill_tokens_per_s) {
                prefillTimeSec = (1 / usage.extra.prefill_tokens_per_s) * usage.prompt_tokens;
            }
            if (usage.extra.decode_tokens_per_s) {
                decodeTokensPerSec = usage.extra.decode_tokens_per_s;
            }
        }
    }

    const totalDurationSec = (performance.now() - runStartTime) / 1000;
    const memAfter = getMemoryUsage();
    const tokenStats = calculatePercentiles(tokenLatencies);

    return {
        runIndex,
        promptLength,
        totalPromptTokens: totalPromptTokens || promptLength,
        prefillTimeSec,
        decodeTokensPerSec,
        totalDurationSec,
        tokenLatenciesMs: tokenStats,
        memBefore,
        memAfter
    };
}

async function runSweep(promptLength, iterations = 5) {
    logMessage(`\n======================================================`);
    logMessage(`Starting Sweep: ${promptLength} tokens (${iterations} iterations)`);
    logMessage(`======================================================`);

    const runs = [];

    for (let i = 1; i <= iterations; i++) {
        logMessage(`[Run ${i}/${iterations}] Executing...`);
        try {
            const result = await executeSingleRun(promptLength, i);
            runs.push(result);
            
            const prefillStr = result.prefillTimeSec ? `${result.prefillTimeSec.toFixed(3)}s` : "N/A";
            const decodeStr = result.decodeTokensPerSec ? `${result.decodeTokensPerSec.toFixed(2)} tok/s` : "N/A";
            
            logMessage(`  -> Prefill: ${prefillStr} | Decode: ${decodeStr}`);
            logMessage(`  -> Per-token latency: Median=${result.tokenLatenciesMs.median.toFixed(2)}ms, P95=${result.tokenLatenciesMs.p95.toFixed(2)}ms`);
            if (result.memAfter.jsHeapUsedMB) {
                logMessage(`  -> JS Heap Used: ${result.memAfter.jsHeapUsedMB} MB`);
            }
        } catch (error) {
            logMessage(`  -> Run ${i} failed: ${error.message}`);
            break;
        }
    }

    if (runs.length > 0) {
        const validPrefills = runs.map(r => r.prefillTimeSec).filter(v => v !== null);
        const validDecodes = runs.map(r => r.decodeTokensPerSec).filter(v => v !== null);

        const prefillSummary = calculatePercentiles(validPrefills);
        const decodeSummary = calculatePercentiles(validDecodes);

        logMessage(`\n--- Summary for ${promptLength} Tokens (${runs.length} successful runs) ---`);
        logMessage(`Prefill Latency (s) : Median = ${prefillSummary.median.toFixed(3)} | p95 = ${prefillSummary.p95.toFixed(3)}`);
        logMessage(`Decode Rate (tok/s) : Median = ${decodeSummary.median.toFixed(2)} | p95 = ${decodeSummary.p95.toFixed(2)}`);

        window.profilingResults[promptLength] = {
            prefillSummary,
            decodeSummary,
            runs
        };
    }
}

// UI event listeners
document.getElementById("btn-128").addEventListener("click", () => runSweep(128, 5));
document.getElementById("btn-512").addEventListener("click", () => runSweep(512, 5));
document.getElementById("btn-1024").addEventListener("click", () => runSweep(1024, 5));

document.getElementById("btn-export").addEventListener("click", () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(window.profilingResults, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `webgpu_profiling_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
});

initializeWebLLM();