import torch
import unittest
from dynamic_model_builder import create_model_from_config

class TestDynamicModelBuilder(unittest.TestCase):
    def test_sequential_model(self):
        """Test a simple sequential model (similar to original implementation)"""
        config = {
            "layers": [
                {
                    "id": "input_data",
                    "type": "Input"
                },
                {
                    "id": "conv1",
                    "type": "Conv2d",
                    "params": {
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1
                    },
                    "inputs": ["input_data"]
                },
                {
                    "id": "relu1",
                    "type": "ReLU",
                    "params": {},
                    "inputs": ["conv1"]
                },
                {
                    "id": "pool1",
                    "type": "MaxPool2d",
                    "params": {
                        "kernel_size": 2,
                        "stride": 2
                    },
                    "inputs": ["relu1"]
                }
            ]
        }
        
        model = create_model_from_config(config)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 64, 16, 16))
        
    def test_resnet_block(self):
        """Test a model with skip connection (ResNet-style)"""
        config = {
            "layers": [
                {
                    "id": "input_data",
                    "type": "Input"
                },
                {
                    "id": "conv1",
                    "type": "Conv2d",
                    "params": {
                        "in_channels": 3,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "padding": 1
                    },
                    "inputs": ["input_data"]
                },
                {
                    "id": "relu1",
                    "type": "ReLU",
                    "params": {},
                    "inputs": ["conv1"]
                },
                {
                    "id": "conv2",
                    "type": "Conv2d",
                    "params": {
                        "in_channels": 16,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "padding": 1
                    },
                    "inputs": ["relu1"]
                },
                {
                    "id": "skip_connection_add",
                    "type": "Add",
                    "inputs": ["relu1", "conv2"]
                },
                {
                    "id": "relu2",
                    "type": "ReLU",
                    "params": {},
                    "inputs": ["skip_connection_add"]
                }
            ]
        }
        
        model = create_model_from_config(config)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 16, 32, 32))
        
    def test_mini_unet(self):
        """Test a mini U-Net architecture with skip connections"""
        config = {
            "layers": [
                # Encoder
                {"id": "input_data", "type": "Input"},
                {
                    "id": "enc1",
                    "type": "Conv2d",
                    "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
                    "inputs": ["input_data"]
                },
                {"id": "enc1_relu", "type": "ReLU", "params": {}, "inputs": ["enc1"]},
                {
                    "id": "pool1",
                    "type": "MaxPool2d",
                    "params": {"kernel_size": 2, "stride": 2},
                    "inputs": ["enc1_relu"]
                },
                
                # Bottleneck
                {
                    "id": "bottleneck",
                    "type": "Conv2d",
                    "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1},
                    "inputs": ["pool1"]
                },
                {"id": "bottleneck_relu", "type": "ReLU", "params": {}, "inputs": ["bottleneck"]},
                
                # Decoder
                {
                    "id": "upsample1",
                    "type": "ConvTranspose2d",
                    "params": {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 2,
                        "stride": 2
                    },
                    "inputs": ["bottleneck_relu"]
                },
                {
                    "id": "skip_connection",
                    "type": "Concat",
                    "inputs": ["enc1_relu", "upsample1"]
                },
                {
                    "id": "dec1",
                    "type": "Conv2d",
                    "params": {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "padding": 1},
                    "inputs": ["skip_connection"]
                },
                {"id": "dec1_relu", "type": "ReLU", "params": {}, "inputs": ["dec1"]},
                {
                    "id": "output_conv",
                    "type": "Conv2d",
                    "params": {"in_channels": 64, "out_channels": 3, "kernel_size": 1},
                    "inputs": ["dec1_relu"]
                }
            ]
        }
        
        model = create_model_from_config(config)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 3, 32, 32))

if __name__ == '__main__':
    unittest.main() 