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

class DenseBlock(nn.Module):
    """Dense block for DenseNet"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.ReLU(),
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1)
            )
            self.layers.append(layer)
            current_channels += growth_rate
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class DirectDenseNet(nn.Module):
    """Simple DenseNet implementation"""
    def __init__(self):
        super().__init__()
        self.initial = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=4)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(64 + 4*32),
            nn.ReLU(),
            nn.Conv2d(64 + 4*32, 128, kernel_size=1),
            nn.AvgPool2d(2)
        )
        self.dense2 = DenseBlock(128, growth_rate=32, num_layers=4)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(128 + 4*32),
            nn.ReLU(),
            nn.Conv2d(128 + 4*32, 64, kernel_size=1),
            nn.Upsample(scale_factor=2)
        )
        self.final = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        return self.final(x)

class DirectFPN(nn.Module):
    """Feature Pyramid Network implementation"""
    def __init__(self):
        super().__init__()
        # Bottom-up pathway
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Top-down pathway
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Lateral connections
        self.lateral1 = nn.Conv2d(128, 128, kernel_size=1)
        self.lateral2 = nn.Conv2d(64, 64, kernel_size=1)
        
        self.output = nn.Conv2d(64, 3, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Bottom-up
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(c2))
        
        # Top-down
        p3 = c3
        p2 = self.relu(self.lateral1(c2) + self.upconv1(p3))
        p1 = self.relu(self.lateral2(c1) + self.upconv2(p2))
        
        return self.output(p1)

class SelfAttention(nn.Module):
    """Simple self-attention module"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        q = self.query(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height*width)
        v = self.value(x).view(batch_size, -1, height*width)
        
        attention = torch.bmm(q, k)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

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
    """Create config for dynamic ResUNet directly rather than modifying UNet"""
    return {
        "layers": [
            {"id": "input_data", "type": "Input"},
            
            # Encoder Path - Level 1 with Residual
            {"id": "enc1_conv1", "type": "Conv2d", 
             "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["input_data"]},
            {"id": "enc1_relu1", "type": "ReLU", "params": {}, "inputs": ["enc1_conv1"]},
            {"id": "enc1_conv2", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["enc1_relu1"]},
            {"id": "enc1_relu2", "type": "ReLU", "params": {}, "inputs": ["enc1_conv2"]},
            {"id": "enc1_skip", "type": "Conv2d",
             "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 1},
             "inputs": ["input_data"]},
            {"id": "enc1_add", "type": "Add",
             "inputs": ["enc1_relu2", "enc1_skip"]},
            
            # Pooling and Encoder Level 2 with Residual
            {"id": "pool1", "type": "MaxPool2d",
             "params": {"kernel_size": 2}, 
             "inputs": ["enc1_add"]},
            {"id": "enc2_conv1", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["pool1"]},
            {"id": "enc2_relu1", "type": "ReLU", "params": {}, "inputs": ["enc2_conv1"]},
            {"id": "enc2_conv2", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["enc2_relu1"]},
            {"id": "enc2_relu2", "type": "ReLU", "params": {}, "inputs": ["enc2_conv2"]},
            {"id": "enc2_skip", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 1},
             "inputs": ["pool1"]},
            {"id": "enc2_add", "type": "Add",
             "inputs": ["enc2_relu2", "enc2_skip"]},
            
            # Pooling and Bottleneck with Residual
            {"id": "pool2", "type": "MaxPool2d",
             "params": {"kernel_size": 2},
             "inputs": ["enc2_add"]},
            {"id": "bottleneck_conv1", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "padding": 1},
             "inputs": ["pool2"]},
            {"id": "bottleneck_relu1", "type": "ReLU", "params": {}, "inputs": ["bottleneck_conv1"]},
            {"id": "bottleneck_conv2", "type": "Conv2d",
             "params": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1},
             "inputs": ["bottleneck_relu1"]},
            {"id": "bottleneck_relu2", "type": "ReLU", "params": {}, "inputs": ["bottleneck_conv2"]},
            {"id": "bottleneck_skip", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 1},
             "inputs": ["pool2"]},
            {"id": "bottleneck_add", "type": "Add",
             "inputs": ["bottleneck_relu2", "bottleneck_skip"]},
            
            # Decoder Path
            {"id": "upconv1", "type": "ConvTranspose2d",
             "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 2, "stride": 2},
             "inputs": ["bottleneck_add"]},
            {"id": "concat1", "type": "Concat",
             "inputs": ["upconv1", "enc2_add"]},
            
            # Decoder Level 1 with Residual
            {"id": "dec1_conv1", "type": "Conv2d",
             "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["concat1"]},
            {"id": "dec1_relu1", "type": "ReLU", "params": {}, "inputs": ["dec1_conv1"]},
            {"id": "dec1_conv2", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 1},
             "inputs": ["dec1_relu1"]},
            {"id": "dec1_relu2", "type": "ReLU", "params": {}, "inputs": ["dec1_conv2"]},
            {"id": "dec1_skip", "type": "Conv2d",
             "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 1},
             "inputs": ["concat1"]},
            {"id": "dec1_add", "type": "Add",
             "inputs": ["dec1_relu2", "dec1_skip"]},
            
            # Decoder Level 2 with Residual
            {"id": "upconv2", "type": "ConvTranspose2d",
             "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 2, "stride": 2},
             "inputs": ["dec1_add"]},
            {"id": "concat2", "type": "Concat",
             "inputs": ["upconv2", "enc1_add"]},
            {"id": "dec2_conv1", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["concat2"]},
            {"id": "dec2_relu1", "type": "ReLU", "params": {}, "inputs": ["dec2_conv1"]},
            {"id": "dec2_conv2", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["dec2_relu1"]},
            {"id": "dec2_relu2", "type": "ReLU", "params": {}, "inputs": ["dec2_conv2"]},
            {"id": "dec2_skip", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 1},
             "inputs": ["concat2"]},
            {"id": "dec2_add", "type": "Add",
             "inputs": ["dec2_relu2", "dec2_skip"]},
            
            # Output
            {"id": "outconv", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 3, "kernel_size": 1},
             "inputs": ["dec2_add"]}
        ]
    }

def create_densenet_config():
    """Create config for dynamic DenseNet"""
    return {
        "layers": [
            {"id": "input_data", "type": "Input"},
            
            # Initial convolution
            {"id": "initial", "type": "Conv2d",
             "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3},
             "inputs": ["input_data"]},
            
            # First Dense Block
            {"id": "dense1_bn1", "type": "BatchNorm2d", "params": {"num_features": 64}, "inputs": ["initial"]},
            {"id": "dense1_relu1", "type": "ReLU", "params": {}, "inputs": ["dense1_bn1"]},
            {"id": "dense1_conv1", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 32, "kernel_size": 3, "padding": 1},
             "inputs": ["dense1_relu1"]},
            
            {"id": "dense1_concat1", "type": "Concat", "inputs": ["initial", "dense1_conv1"]},
            {"id": "dense1_bn2", "type": "BatchNorm2d", "params": {"num_features": 96}, "inputs": ["dense1_concat1"]},
            {"id": "dense1_relu2", "type": "ReLU", "params": {}, "inputs": ["dense1_bn2"]},
            {"id": "dense1_conv2", "type": "Conv2d",
             "params": {"in_channels": 96, "out_channels": 32, "kernel_size": 3, "padding": 1},
             "inputs": ["dense1_relu2"]},
            
            {"id": "dense1_concat2", "type": "Concat", "inputs": ["dense1_concat1", "dense1_conv2"]},
            
            # Transition 1
            {"id": "trans1_bn", "type": "BatchNorm2d", "params": {"num_features": 128}, "inputs": ["dense1_concat2"]},
            {"id": "trans1_relu", "type": "ReLU", "params": {}, "inputs": ["trans1_bn"]},
            {"id": "trans1_conv", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 1},
             "inputs": ["trans1_relu"]},
            {"id": "trans1_pool", "type": "AvgPool2d", "params": {"kernel_size": 2}, "inputs": ["trans1_conv"]},
            
            # Second Dense Block
            {"id": "dense2_bn1", "type": "BatchNorm2d", "params": {"num_features": 128}, "inputs": ["trans1_pool"]},
            {"id": "dense2_relu1", "type": "ReLU", "params": {}, "inputs": ["dense2_bn1"]},
            {"id": "dense2_conv1", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 32, "kernel_size": 3, "padding": 1},
             "inputs": ["dense2_relu1"]},
            
            {"id": "dense2_concat1", "type": "Concat", "inputs": ["trans1_pool", "dense2_conv1"]},
            
            # Transition 2 (Upsampling)
            {"id": "trans2_bn", "type": "BatchNorm2d", "params": {"num_features": 160}, "inputs": ["dense2_concat1"]},
            {"id": "trans2_relu", "type": "ReLU", "params": {}, "inputs": ["trans2_bn"]},
            {"id": "trans2_conv", "type": "Conv2d",
             "params": {"in_channels": 160, "out_channels": 64, "kernel_size": 1},
             "inputs": ["trans2_relu"]},
            {"id": "trans2_up", "type": "Upsample", "params": {"scale_factor": 2}, "inputs": ["trans2_conv"]},
            
            # Final convolution
            {"id": "final", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 3, "kernel_size": 1},
             "inputs": ["trans2_up"]}
        ]
    }

def create_fpn_config():
    """Create config for dynamic FPN"""
    return {
        "layers": [
            {"id": "input_data", "type": "Input"},
            
            # Bottom-up pathway
            {"id": "conv1", "type": "Conv2d",
             "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
             "inputs": ["input_data"]},
            {"id": "relu1", "type": "ReLU", "params": {}, "inputs": ["conv1"]},
            
            {"id": "conv2", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1},
             "inputs": ["relu1"]},
            {"id": "relu2", "type": "ReLU", "params": {}, "inputs": ["conv2"]},
            
            {"id": "conv3", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1},
             "inputs": ["relu2"]},
            {"id": "relu3", "type": "ReLU", "params": {}, "inputs": ["conv3"]},
            
            # Top-down pathway
            {"id": "upconv1", "type": "ConvTranspose2d",
             "params": {"in_channels": 256, "out_channels": 128, "kernel_size": 2, "stride": 2},
             "inputs": ["relu3"]},
            
            {"id": "lateral1", "type": "Conv2d",
             "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 1},
             "inputs": ["relu2"]},
            
            {"id": "add1", "type": "Add", "inputs": ["upconv1", "lateral1"]},
            {"id": "relu4", "type": "ReLU", "params": {}, "inputs": ["add1"]},
            
            {"id": "upconv2", "type": "ConvTranspose2d",
             "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 2, "stride": 2},
             "inputs": ["relu4"]},
            
            {"id": "lateral2", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 1},
             "inputs": ["relu1"]},
            
            {"id": "add2", "type": "Add", "inputs": ["upconv2", "lateral2"]},
            {"id": "relu5", "type": "ReLU", "params": {}, "inputs": ["add2"]},
            
            {"id": "output", "type": "Conv2d",
             "params": {"in_channels": 64, "out_channels": 3, "kernel_size": 1},
             "inputs": ["relu5"]}
        ]
    }

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
    print("Creant models...")
    
    # Crear tots els models
    models = {
        "UNet (Direct)": DirectUNet(),
        "UNet (Dynamic)": create_model_from_config(create_unet_config()),
        "ResUNet (Direct)": DirectResUNet(),
        "ResUNet (Dynamic)": create_model_from_config(create_resunet_config()),
        "DenseNet (Direct)": DirectDenseNet(),
        "DenseNet (Dynamic)": create_model_from_config(create_densenet_config()),
        "FPN (Direct)": DirectFPN(),
        "FPN (Dynamic)": create_model_from_config(create_fpn_config()),
    }
    
    # Crear tensors d'entrada de diferents mides
    input_sizes = [(1, 3, 256, 256), (1, 3, 512, 512)]
    results = {}
    
    for size in input_sizes:
        print(f"\nExecutant benchmarks per a mida d'entrada {size}...")
        x = torch.randn(*size)
        results[str(size)] = {}
        
        for name, model in models.items():
            results[str(size)][name] = benchmark_model(model, x, num_iterations=50, name=name)
    
    # Imprimir resultats en format de taula
    print("\nResultats Detallats:")
    print("-" * 120)
    print(f"{'Input Size':<15} {'Model Type':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Overhead %':<12}")
    print("-" * 120)
    
    for size in input_sizes:
        size_results = results[str(size)]
        
        # Calcular overhead per a cada tipus de model
        model_types = ["UNet", "ResUNet", "DenseNet", "FPN"]
        
        for model_type in model_types:
            direct_name = f"{model_type} (Direct)"
            dynamic_name = f"{model_type} (Dynamic)"
            
            direct_time = size_results[direct_name][0]  # Avg time
            dynamic_time = size_results[dynamic_name][0]  # Avg time
            overhead = ((dynamic_time - direct_time) / direct_time) * 100
            
            # Imprimir resultats per al model directe
            print(f"{str(size):<15} {direct_name:<25} "
                  f"{size_results[direct_name][0]:>11.3f} "
                  f"{size_results[direct_name][1]:>11.3f} "
                  f"{size_results[direct_name][2]:>11.3f} {'N/A':>11}")
            
            # Imprimir resultats per al model dinàmic
            print(f"{'':<15} {dynamic_name:<25} "
                  f"{size_results[dynamic_name][0]:>11.3f} "
                  f"{size_results[dynamic_name][1]:>11.3f} "
                  f"{size_results[dynamic_name][2]:>11.3f} "
                  f"{overhead:>11.1f}")
            
            print("-" * 120)
    
    # Anàlisi estadística
    print("\nAnàlisi Estadística:")
    print("-" * 50)
    
    total_overhead = []
    for size in input_sizes:
        size_results = results[str(size)]
        for model_type in model_types:
            direct_name = f"{model_type} (Direct)"
            dynamic_name = f"{model_type} (Dynamic)"
            
            overhead = ((size_results[dynamic_name][0] - size_results[direct_name][0]) 
                       / size_results[direct_name][0] * 100)
            total_overhead.append(overhead)
    
    avg_overhead = sum(total_overhead) / len(total_overhead)
    min_overhead = min(total_overhead)
    max_overhead = max(total_overhead)
    
    print(f"Overhead mitjà: {avg_overhead:.1f}%")
    print(f"Overhead mínim: {min_overhead:.1f}%")
    print(f"Overhead màxim: {max_overhead:.1f}%") 