# Smart Dispatch ✨

Smart Dispatch is an edge-native Chrome Extension that dynamically scans email drafts in Gmail using an on-device quantized ONNX model to suggest contextual file attachments and links based on pure semantic intent.

## Features
- **Local-First & Zero Telemetry:** All inference runs directly in your browser using Transformers.js and WebGPU/WASM. No email data ever leaves your device.
- **Pure Semantic Intent:** Utilizes a highly optimized `all-MiniLM-L6-v2` embedding model to detect abstract contextual intents, not just exact keywords.
- **Semantic Chunking:** Analyzes long emails by breaking them into contextually accurate chunks, preventing "context dilution".
- **Dynamic Asset Injection:** Surfaces a slick drag-and-drop modal inside Gmail to instantly inject relevant links and files directly into your draft.
- **IndexedDB Vector Store:** Securely stores your asset embeddings locally.

## Performance Benchmarks
By dynamically quantizing our fine-tuned PyTorch model into INT8 ONNX, we achieved massive performance gains suitable for edge-device execution:
- **Model Footprint:** 87.08 MB (FP32) ➔ **22.82 MB (INT8)** *(3.82x Reduction)*
- **Single-Pass Embedding Latency:** 4.99 ms (FP32) ➔ **1.47 ms (INT8)** *(3.40x Speedup)*
- **Memory Overhead:** Runs under 60MB heap in the Chrome background offscreen document, preventing tab eviction.

## Installation Instructions

1. Clone or download this repository.
2. Open Google Chrome and navigate to `chrome://extensions/`.
3. Toggle **Developer mode** ON in the top right corner.
4. Click **Load unpacked** and select the `extension/` directory.
5. The Smart Dispatch puzzle piece icon will appear in your toolbar!

## Usage
1. Click the Smart Dispatch icon in your toolbar to open the Asset Manager.
2. Register your commonly used links and files (e.g., your resume, portfolio links, project demos).
3. Open **Gmail** and start a new draft.
4. Type an email with an implicit intent to attach something (e.g., *"I've attached my resume and the music player demo for your review."*).
5. Click the draggable **✨ Analyze Draft** button injected into the Gmail compose window.
6. Select your matched assets from the popup modal and click **Proceed** to inject them directly into your email!

## Architecture Highlights
- **ML Pipeline:** Synthetic contrastive triplets generated via Gemini 1.5 Flash. Model trained using `MultipleNegativesRankingLoss` (Sentence Transformers) and exported via `optimum-cli`.
- **Extension Runtime:** Vanilla JS and Tailwind CSS. Employs a robust MV3 architecture tunneling inference requests from the Content Script, through the Background Worker, into an Offscreen Document for unrestricted WebGPU/WASM thread execution.
