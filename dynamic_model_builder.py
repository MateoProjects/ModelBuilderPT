import torch
import torch.nn as nn
from typing import Dict, List, Any, Union
import json

class OperationType:
    """Custom operations that are not direct PyTorch layers"""
    ADD = "Add"
    CONCAT = "Concat"
    INPUT = "Input"

class LayerOperations:
    """Handler for custom operations between layers"""
    
    @staticmethod
    def add(inputs: List[torch.Tensor]) -> torch.Tensor:
        """Add multiple tensors element-wise"""
        return torch.add(*inputs) if len(inputs) == 2 else sum(inputs)
    
    @staticmethod
    def concat(inputs: List[torch.Tensor], dim: int = 1) -> torch.Tensor:
        """Concatenate multiple tensors along specified dimension"""
        return torch.cat(inputs, dim=dim)


class DynamicModel(nn.Module):
    def __init__(self, config: Union[str, dict, List[dict]]):
        """
        Initialize the dynamic PyTorch model.
        
        Args:
            config: Either a path to JSON config file, a config dict, or a list of layer configurations
        """
        super(DynamicModel, self).__init__()
        
        # Load configuration
        if isinstance(config, str):
            with open(config) as f:
                self.config = json.load(f)
        elif isinstance(config, list):
            self.config = {"layers": config}
        else:
            self.config = config
            
        # Store layers in ModuleDict for easy access by ID
        self.layers = nn.ModuleDict()
        self.custom_ops = {
            OperationType.ADD: LayerOperations.add,
            OperationType.CONCAT: LayerOperations.concat
        }
        

        
        self._create_layers()
        self.output_layers = self.config.get("output_layers", [self.config["layers"][-1]["id"]])
        
    def _create_layers(self):
        """Create all layers defined in the configuration"""
        for layer_config in self.config["layers"]:
            layer_id = layer_config["id"]
            layer_type = layer_config["type"]
            
            if layer_type == OperationType.INPUT:
                continue
                
            if layer_type in [OperationType.ADD, OperationType.CONCAT]:
                continue

            layer_params = layer_config.get("params", {})
            try:
                layer = getattr(nn, layer_type)(**layer_params)
                self.layers[layer_id] = layer
            except Exception as e:
                raise ValueError(f"Error creating layer {layer_id} of type {layer_type}: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        
        outputs = {"input_data": x}
        
        # Process each layer in order
        for layer_config in self.config["layers"]:
            layer_id = layer_config["id"]
            layer_type = layer_config["type"]
            
            if layer_id == "input_data":
                continue
            
            # Get input tensors for this layer
            input_tensors = [outputs[input_id] for input_id in layer_config["inputs"]]
            
            # Process based on layer type
            if layer_type in self.custom_ops:
                outputs[layer_id] = self.custom_ops[layer_type](input_tensors)
            elif layer_type != OperationType.INPUT:
                if len(input_tensors) == 1:
                    outputs[layer_id] = self.layers[layer_id](input_tensors[0])
                else:
                    raise ValueError(f"Layer {layer_id} expects 1 input but got {len(input_tensors)}")
        
        # Return outputs from specified output layers
        if len(self.output_layers) == 1:
            return outputs[self.output_layers[0]]
        return [outputs[layer_id] for layer_id in self.output_layers]

def create_model_from_config(model_config: Union[str, dict, List[dict]]) -> DynamicModel:
    """
    Create a PyTorch model from a configuration
    
    Args:
        model_config: Either path to JSON config file, a config dict, or a list of layer configurations
        
    Returns:
        Instantiated DynamicModel
    """
    return DynamicModel(model_config) 