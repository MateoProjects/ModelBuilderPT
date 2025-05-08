import torch
import torch.nn as nn
import time
from dynamic_model_builder import create_model_from_config

class DoubleConv(nn.Module):
    """(Conv2d -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class DirectUNet(nn.Module):
    """Standard UNet implementation"""
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)  # 256 perquè concatenem amb skip connection
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)   # 128 perquè concatenem amb skip connection
        
        # Output
        self.outconv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv1(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)
        
        return self.outconv(x)

class ResBlock(nn.Module):
    """Residual block for ResUNet"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = DoubleConv(in_channels, out_channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv_block(x) + self.skip_conv(x)

class DirectResUNet(nn.Module):
    """ResUNet implementation with residual connections"""
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = ResBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock(128, 256)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ResBlock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResBlock(128, 64)
        
        # Output
        self.outconv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv1(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)
        
        return self.outconv(x)

def create_unet_config():
    """Create config for dynamic UNet"""
    return {
        "layers": [
            {"id": "input_data", "type": "Input"},
            
            # Encoder Path
            {"id": "enc1_conv1", "type": "Conv2d", 
             "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["input_data"]},
            {"id": "enc1_relu1", "type": "ReLU", "params": {}, "inputs": ["enc1_conv1"]},
            {"id": "enc1_conv2", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["enc1_relu1"]},
            {"id": "enc1_relu2", "type": "ReLU", "params": {}, "inputs": ["enc1_conv2"]},
            
            {"id": "pool1", "type": "MaxPool2d",
             "params": {"kernel_size": 2}, "inputs": ["enc1_relu2"]},
            
            {"id": "enc2_conv1", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["pool1"]},
            {"id": "enc2_relu1", "type": "ReLU", "params": {}, "inputs": ["enc2_conv1"]},
            {"id": "enc2_conv2", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["enc2_relu1"]},
            {"id": "enc2_relu2", "type": "ReLU", "params": {}, "inputs": ["enc2_conv2"]},
            
            {"id": "pool2", "type": "MaxPool2d",
             "params": {"kernel_size": 2}, "inputs": ["enc2_relu2"]},
            
            # Bottleneck
            {"id": "bottleneck_conv1", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "padding": 1},
             "inputs": ["pool2"]},
            {"id": "bottleneck_relu1", "type": "ReLU", "params": {}, "inputs": ["bottleneck_conv1"]},
            {"id": "bottleneck_conv2", "type": "Conv2d",
             "params": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1},
             "inputs": ["bottleneck_relu1"]},
            {"id": "bottleneck_relu2", "type": "ReLU", "params": {}, "inputs": ["bottleneck_conv2"]},
            
            # Decoder Path
            {"id": "upconv1", "type": "ConvTranspose2d",
             "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 2, "stride": 2},
             "inputs": ["bottleneck_relu2"]},
            
            {"id": "concat1", "type": "Concat",
             "inputs": ["upconv1", "enc2_relu2"]},
            
            {"id": "dec1_conv1", "type": "Conv2d",
             "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["concat1"]},
            {"id": "dec1_relu1", "type": "ReLU", "params": {}, "inputs": ["dec1_conv1"]},
            {"id": "dec1_conv2", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["dec1_relu1"]},
            {"id": "dec1_relu2", "type": "ReLU", "params": {}, "inputs": ["dec1_conv2"]},
            
            {"id": "upconv2", "type": "ConvTranspose2d",
             "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 2, "stride": 2},
             "inputs": ["dec1_relu2"]},
            
            {"id": "concat2", "type": "Concat",
             "inputs": ["upconv2", "enc1_relu2"]},
            
            {"id": "dec2_conv1", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["concat2"]},
            {"id": "dec2_relu1", "type": "ReLU", "params": {}, "inputs": ["dec2_conv1"]},
            {"id": "dec2_conv2", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["dec2_relu1"]},
            {"id": "dec2_relu2", "type": "ReLU", "params": {}, "inputs": ["dec2_conv2"]},
            
            # Output
            {"id": "outconv", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 3, "kernel_size": 1},
             "inputs": ["dec2_relu2"]}
        ]
    }

def create_resunet_config():
    """Create config for dynamic ResUNet"""
    config = create_unet_config()
    
    # Primer eliminem les capes que volem modificar
    config["layers"] = [layer for layer in config["layers"] if layer["id"] not in 
                       ["pool1", "pool2", "upconv1", "concat1", "upconv2", "concat2", "outconv"]]
    
    # Afegim les capes residuals i les capes modificades en l'ordre correcte
    config["layers"].extend([
        # Encoder Path Residual Connections - Level 1
        {"id": "enc1_skip", "type": "Conv2d",
         "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 1},
         "inputs": ["input_data"]},
        {"id": "enc1_add", "type": "Add",
         "inputs": ["enc1_relu2", "enc1_skip"]},
         
        # Pooling and Encoder Level 2
        {"id": "pool1_op", "type": "MaxPool2d",
         "params": {"kernel_size": 2}, 
         "inputs": ["enc1_add"]},
        
        {"id": "enc2_skip", "type": "Conv2d",
         "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 1},
         "inputs": ["pool1_op"]},
        {"id": "enc2_add", "type": "Add",
         "inputs": ["enc2_relu2", "enc2_skip"]},
         
        # Pooling and Bottleneck
        {"id": "pool2_op", "type": "MaxPool2d",
         "params": {"kernel_size": 2},
         "inputs": ["enc2_add"]},
         
        {"id": "bottleneck_skip", "type": "Conv2d",
         "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 1},
         "inputs": ["pool2_op"]},
        {"id": "bottleneck_add", "type": "Add",
         "inputs": ["bottleneck_relu2", "bottleneck_skip"]},
         
        # Decoder Path
        {"id": "upconv1", "type": "ConvTranspose2d",
         "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 2, "stride": 2},
         "inputs": ["bottleneck_add"]},
         
        {"id": "concat1", "type": "Concat",
         "inputs": ["upconv1", "enc2_add"]},
         
        {"id": "dec1_skip", "type": "Conv2d",
         "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 1},
         "inputs": ["concat1"]},
        {"id": "dec1_add", "type": "Add",
         "inputs": ["dec1_relu2", "dec1_skip"]},
         
        {"id": "upconv2", "type": "ConvTranspose2d",
         "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 2, "stride": 2},
         "inputs": ["dec1_add"]},
         
        {"id": "concat2", "type": "Concat",
         "inputs": ["upconv2", "enc1_add"]},
         
        {"id": "dec2_skip", "type": "Conv2d",
         "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 1},
         "inputs": ["concat2"]},
        {"id": "dec2_add", "type": "Add",
         "inputs": ["dec2_relu2", "dec2_skip"]},
         
        # Output
        {"id": "outconv", "type": "Conv2d",
         "params": {"in_channels": 64, "out_channels": 3, "kernel_size": 1},
         "inputs": ["dec2_add"]}
    ])
    
    return config

def benchmark_model(model, input_tensor, num_iterations=100, name="Unknown"):
    """Benchmark model forward pass"""
    print(f"\nBenchmarking {name}...")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Number of iterations: {num_iterations}")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        _ = model(input_tensor)
    
    # Benchmark
    print("Running benchmark...")
    times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        _ = model(input_tensor)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return avg_time, min_time, max_time

if __name__ == "__main__":
    print("Creating models...")
    
    # Create all models
    unet_direct = DirectUNet()
    resunet_direct = DirectResUNet()
    unet_dynamic = create_model_from_config(create_unet_config())
    resunet_dynamic = create_model_from_config(create_resunet_config())
    
    # Create input tensor (batch_size=1, channels=3, height=256, width=256)
    x = torch.randn(1, 3, 256, 256)
    
    # Benchmark all models
    results = {}
    
    for name, model in [
        ("UNet (Direct)", unet_direct),
        ("UNet (Dynamic)", unet_dynamic),
        ("ResUNet (Direct)", resunet_direct),
        ("ResUNet (Dynamic)", resunet_dynamic)
    ]:
        results[name] = benchmark_model(model, x, num_iterations=50, name=name)
    
    # Print comparison
    print("\nResults Comparison:")
    print(f"{'Model Type':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 56)
    
    for name, (avg, min_t, max_t) in results.items():
        print(f"{name:<20} {avg:>11.3f} {min_t:>11.3f} {max_t:>11.3f}")
    
    # Calculate overheads
    unet_overhead = ((results["UNet (Dynamic)"][0] - results["UNet (Direct)"][0]) 
                    / results["UNet (Direct)"][0] * 100)
    resunet_overhead = ((results["ResUNet (Dynamic)"][0] - results["ResUNet (Direct)"][0]) 
                       / results["ResUNet (Direct)"][0] * 100)
    
    print("\nOverhead Analysis:")
    print(f"UNet Dynamic Overhead: {unet_overhead:.1f}%")
    print(f"ResUNet Dynamic Overhead: {resunet_overhead:.1f}%") 