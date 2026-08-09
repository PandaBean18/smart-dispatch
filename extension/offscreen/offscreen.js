import { pipeline, env } from '../lib/transformers.min.js';

// Configure environment
// Disable loading local models from the file system (not supported in browser)
env.allowLocalModels = false;
// Allow using the cache API
env.useBrowserCache = true;

class InferenceEngine {
    static instance = null;
    static profiling = {
        coldStartTime: null,
        totalRamLoaded: null
    };

    static async getInstance() {
        if (!this.instance) {
            const startTime = performance.now();
            
            // We use 'feature-extraction' for MiniLM to get the dense vectors
            // WebGPU execution provider explicitly requested with fallback to WASM
            this.instance = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
                device: 'webgpu', // Explicit WebGPU Support
                quantized: true,  // Use INT8
            });
            
            this.profiling.coldStartTime = performance.now() - startTime;
            
            if (performance.memory) {
                this.profiling.totalRamLoaded = performance.memory.usedJSHeapSize / (1024 * 1024);
            }
            
            console.log(`[Profiler] Cold Start Load Time: ${this.profiling.coldStartTime.toFixed(2)}ms`);
            console.log(`[Profiler] Peak Heap RAM used: ${this.profiling.totalRamLoaded ? this.profiling.totalRamLoaded.toFixed(2) + 'MB' : 'N/A'}`);
        }
        return this.instance;
    }
}

// Vector math: Pure Semantic Retrieval (Cosine Similarity)
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] ** 2;
        normB += vecB[i] ** 2;
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// IndexedDB Helper for Storing Embeddings locally
class VectorStore {
    static async getDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open("SmartDispatchStore", 1);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains("assets")) {
                    db.createObjectStore("assets", { keyPath: "id", autoIncrement: true });
                }
            };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    static async saveAsset(asset) {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction("assets", "readwrite");
            const store = tx.objectStore("assets");
            store.add(asset);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    static async getAllAssets() {
        const db = await this.getDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction("assets", "readonly");
            const store = tx.objectStore("assets");
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }
}

// Helper to intelligently chunk long email drafts
function chunkText(text) {
  if (!text || typeof text !== 'string') return [];

  // 1. Split into paragraphs or logical line blocks first
  const blocks = text.split(/\n+/).map(b => b.trim()).filter(Boolean);
  const chunks = [];

  for (const block of blocks) {
    // If paragraph is under ~200 characters, treat the whole paragraph as one chunk
    if (block.length <= 200) {
      if (block.length >= 10) chunks.push(block);
      continue;
    }

    // 2. For longer paragraphs, split by sentence boundaries
    const sentences = block
      .split(/(?<=[.?!;])\s+/)
      .map(s => s.trim())
      .filter(s => s.length >= 10);

    chunks.push(...sentences);
  }

  // Fallback: If no valid chunks found, return original text if >= 10 chars
  return chunks.length > 0 ? chunks : (text.length >= 10 ? [text] : []);
}

// Listen for messages from the background service worker
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'INFERENCE_REQUEST') {
        (async () => {
            const startInference = performance.now();
            const extractor = await InferenceEngine.getInstance();

            
            const chunks = chunkText(message.text);
            if (chunks.length === 0) {
                sendResponse({ success: true, suggestions: [], profiler: {} });
                return;
            }
            
            // Extract feature vector for ALL chunks simultaneously (Batch Processing)
            const output = await extractor(chunks, { pooling: 'mean', normalize: true });
            
            // Safe extraction of the batched 2D tensor into individual JS Arrays
            const numChunks = output.dims[0];
            const vecSize = output.dims[1];
            
            const chunkEmbeddings = [];
            for (let i = 0; i < numChunks; i++) {
                const startIdx = i * vecSize;
                const endIdx = startIdx + vecSize;
                // use .subarray for TypedArray (Float32Array)
                chunkEmbeddings.push(Array.from(output.data.subarray(startIdx, endIdx)));
            }
            
            const inferenceTime = performance.now() - startInference;
            console.log(`[Profiler] Batch Embedding Latency (${chunks.length} chunks): ${inferenceTime.toFixed(2)}ms`);

            // Fetch all stored assets
            const assets = await VectorStore.getAllAssets();
            
            // Compute Max-Pooled cosine similarity for each asset
            const results = assets.map(asset => {
                let maxScore = -1;
                for (const chunkVec of chunkEmbeddings) {
                    const sim = cosineSimilarity(chunkVec, asset.embedding);
                    if (sim > maxScore) {
                        maxScore = sim;
                    }
                }
                return { ...asset, score: maxScore };
            });
            
            // Sort by highest score first and take Top-5
            results.sort((a, b) => b.score - a.score);
            const topK = results.slice(0, 5);
            
            sendResponse({ 
                success: true, 
                suggestions: topK, 
                profiler: {
                    latencyMs: inferenceTime,
                    coldStartMs: InferenceEngine.profiling.coldStartTime
                }
            });
        })();
        return true;
    }
    
    if (message.type === 'EMBED_ASSET_REQUEST') {
        (async () => {
            const extractor = await InferenceEngine.getInstance();
            const textToEmbed = `${message.asset.label} ${message.asset.keywords}`;
            
            const output = await extractor(textToEmbed, { pooling: 'mean', normalize: true });
            const embedding = Array.from(output.data);
            
            const fullAsset = { ...message.asset, embedding };
            await VectorStore.saveAsset(fullAsset);
            
            sendResponse({ success: true });
        })();
        return true;
    }
});
