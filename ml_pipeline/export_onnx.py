import os
import time
import argparse
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def benchmark_model(model, tokenizer, num_runs=100):
    # Prepare dummy inputs simulating an implicit trigger phrase
    dummy_text = "Let me know if you need to review my previous work"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
    
    # Warmup
    for _ in range(10):
        _ = model(**inputs)
        
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model(**inputs)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to ms
        
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    
    return p50, p90, p99

def main():
    parser = argparse.ArgumentParser(description="Export and Quantize Model to ONNX INT8 with Benchmarking")
    parser.add_argument("--model_id", type=str, default="models/smart-dispatch-minilm-v1", help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="models/onnx", help="Output directory for ONNX models")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_id):
        print(f"Warning: Model path {args.model_id} does not exist. Using base 'sentence-transformers/all-MiniLM-L6-v2' for demonstration.")
        args.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        
    fp32_dir = Path(args.output_dir) / "fp32"
    int8_dir = Path(args.output_dir) / "int8"
    
    os.makedirs(fp32_dir, exist_ok=True)
    os.makedirs(int8_dir, exist_ok=True)
    
    print(f"=== Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(int8_dir)
    
    print("\n=== Exporting to ONNX (FP32) ===")
    # Exporting the model using Optimum
    ort_model = ORTModelForFeatureExtraction.from_pretrained(args.model_id, export=True)
    ort_model.save_pretrained(fp32_dir)
    fp32_size = get_dir_size(fp32_dir) / (1024 * 1024)
    print(f"FP32 Model saved to {fp32_dir} (Size: {fp32_size:.2f} MB)")
    
    print("\n=== Quantizing to ONNX (INT8) ===")
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=int8_dir, quantization_config=qconfig)
    
    int8_size = get_dir_size(int8_dir) / (1024 * 1024)
    print(f"INT8 Model saved to {int8_dir} (Size: {int8_size:.2f} MB)")
    print(f"Size Reduction: {fp32_size / int8_size:.2f}x")
    
    print("\n=== Benchmarking Inference Latency (FP32 vs INT8) ===")
    # Load back both models for inference benchmarking
    fp32_model = ORTModelForFeatureExtraction.from_pretrained(fp32_dir)
    int8_model = ORTModelForFeatureExtraction.from_pretrained(int8_dir)
    
    print("Benchmarking FP32 model...")
    fp32_p50, fp32_p90, fp32_p99 = benchmark_model(fp32_model, tokenizer)
    print(f"FP32 Latency (ms): p50={fp32_p50:.2f}, p90={fp32_p90:.2f}, p99={fp32_p99:.2f}")
    
    print("Benchmarking INT8 model...")
    int8_p50, int8_p90, int8_p99 = benchmark_model(int8_model, tokenizer)
    print(f"INT8 Latency (ms): p50={int8_p50:.2f}, p90={int8_p90:.2f}, p99={int8_p99:.2f}")
    
    speedup = fp32_p50 / int8_p50
    print(f"\nSpeedup (p50): {speedup:.2f}x")
    
    # Save benchmark report
    report = f"""
    # Quantization Benchmarks (Python ORT)
    - FP32 Size: {fp32_size:.2f} MB
    - INT8 Size: {int8_size:.2f} MB (Reduction: {fp32_size / int8_size:.2f}x)
    - FP32 Latency p50: {fp32_p50:.2f} ms
    - INT8 Latency p50: {int8_p50:.2f} ms (Speedup: {speedup:.2f}x)
    """
    
    with open(Path(args.output_dir) / "benchmark_report.txt", "w") as f:
        f.write(report)
        
    print("\nBenchmark report saved to models/onnx/benchmark_report.txt")

if __name__ == "__main__":
    main()
