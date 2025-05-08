import torch
import time
from dynamic_model_builder import create_model_from_config

def benchmark_model(model, input_tensor, num_iterations=100):
    """Benchmark model forward pass"""
    print("Starting benchmark...")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Number of iterations: {num_iterations}")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        _ = model(input_tensor)
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    
    for i in range(num_iterations):
        _ = model(input_tensor)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time * 1000  # Convert to milliseconds

if __name__ == "__main__":
    print("Creating model configuration...")
    # Test with a simpler model first
    simple_config = {
        "layers": [
            {"id": "input_data", "type": "Input"},
            {
                "id": "conv1",
                "type": "Conv2d",
                "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
                "inputs": ["input_data"]
            },
            {
                "id": "relu1",
                "type": "ReLU",
                "params": {},
                "inputs": ["conv1"]
            }
        ]
    }
    
    print("Creating model...")
    model = create_model_from_config(simple_config)
    print("Model created successfully")
    
    print("Creating input tensor...")
    x = torch.randn(1, 3, 64, 64)  # Smaller input size
    print("Input tensor created")
    
    # Run benchmark
    print("\nStarting benchmark measurement...")
    avg_time = benchmark_model(model, x)
    print(f"\nResults:")
    print(f"Average forward pass time: {avg_time:.3f} ms") 