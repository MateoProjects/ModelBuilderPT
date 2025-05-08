import torch.nn as nn
import json

def create_layers(config):
    """
    Create layers for a PyTorch model based on the configuration.

    Args:
        config (list): The list of layer configurations.

    Returns:
        nn.Sequential: The PyTorch model layers.
    """
    layers = []
    for layer in config:
        layer_type = layer["type"]

        match layer_type:
            case "Conv1d" | "Conv2d" | "Conv3d":
                ConvLayer = getattr(nn, layer_type)
                layers.append(ConvLayer(
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    padding=layer.get("padding", 0)
                ))

            case "MaxPool1d" | "MaxPool2d" | "MaxPool3d":
                MaxPoolLayer = getattr(nn, layer_type)
                layers.append(MaxPoolLayer(
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    padding=layer.get("padding", 0)
                ))

            case "MaxUnpool1d" | "MaxUnpool2d" | "MaxUnpool3d":
                MaxUnPoolLayer = getattr(nn, layer_type)
                layers.append(MaxUnPoolLayer(
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    padding=layer.get("padding", 0)
                ))

            case "AvgPool1d" | "AvgPool2d" | "AvgPool3d":
                AvgPoolLayer = getattr(nn, layer_type)
                layers.append(AvgPoolLayer(
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", None),
                    padding=layer.get("padding", 0),
                    ceil_mode=layer.get("ceil_mode", False),
                    count_include_pad=layer.get("count_include_pad", True)
                ))

            case "Flatten":
                layers.append(nn.Flatten())

            case "Linear":
                layers.append(nn.Linear(
                    in_features=layer["in_features"],
                    out_features=layer["out_features"]
                ))

            case "ConvTranspose1d" | "ConvTranspose2d" | "ConvTranspose3d":
                ConvLayer = getattr(nn, layer_type)
                layers.append(ConvLayer(
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    padding=layer.get("padding", 0),
                    output_padding=layer.get("output_padding", 0)
                ))

            case "ELU":
                layers.append(nn.ELU(alpha=layer.get("alpha", 1)))

            case "ReLU":
                layers.append(nn.ReLU())

            case "LeakyReLU":
                layers.append(nn.LeakyReLU(negative_slope=layer.get("negative_slope", 0.01)))

            case "GELU":
                layers.append(nn.GELU(approximate=layer.get("approximate", 'none')))

            case "Tanh":
                layers.append(nn.Tanh())

            case "Sigmoid":
                layers.append(nn.Sigmoid())

            case "Softmax":
                layers.append(nn.Softmax(dim=layer.get("dim", None)))

            case "Softmax2d":
                layers.append(nn.Softmax2d())

            case "BatchNorm1d" | "BatchNorm2d" | "BatchNorm3d":
                BatchNorm = getattr(nn, layer_type)
                layers.append(BatchNorm(
                    num_features=layer["num_features"],
                    eps=layer.get("eps", 1e-05),
                    momentum=layer.get("momentum", 0.1),
                    affine=layer.get("affine", True),
                ))

            case "LSTM" | "GRU":
                rnn = getattr(nn, layer_type)
                layers.append(rnn(
                    input_size=layer["input_size"],
                    hidden_size=layer["hidden_size"],
                    num_layers=layer["num_layers"],
                    bias=layer.get("bias", True),
                    batch_first=layer.get("batch_first", False),
                    dropout=layer.get("dropout", 0.0),
                    bidirectional=layer.get("bidirectional", False)
                ))

            case "Dropout" | "Dropout1d" | "Dropout2d" | "Dropout3d":
                dropout = getattr(nn, layer_type)
                layers.append(dropout(
                    p=layer["p"]
                ))

            case _:
                raise ValueError(f"Unknown layer type: {layer_type}")

    return nn.Sequential(*layers)

class DynamicModel(nn.Module):
    def __init__(self, config):
        """
        Initialize the dynamic PyTorch model.

        Args:
            config (list): The list of layer configurations.
        """
        super(DynamicModel, self).__init__()
        self.layers = create_layers(config)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.layers(x)

def create_model_from_config(model_config):
    """
    Create a PyTorch model from a configuration file.

    Args:
        model_config (str or list): The path to the configuration file or the configuration list.

    Returns:
        DynamicModel: The PyTorch model.
    """
    if type(model_config) is not list:
        with open(model_config) as f:
            model_config = json.load(f)
    
    model = DynamicModel(model_config)
    return model
