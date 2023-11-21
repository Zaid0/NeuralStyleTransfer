import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(nn.Module):
    """
    Customized ResNet model for neural style transfer.

    Parameters:
    - content_layer (str, optional): Name of the content layer. Default is 'layer1'.
    - style_layers (list, optional): List of names of style layers. Default is ['layer1', 'layer2'].
    - is_relu (bool, optional): If True, uses ReLU activation after each layer. Default is True.
    - avg_pooling (bool, optional): If True, replaces max pooling with average pooling. Default is True.

    Example:
     model = CustomResNet(content_layer='layer2', style_layers=['layer3', 'layer4'])
    content_features, style_features = model(input_tensor, is_generated=True)
    """

    def __init__(self, content_layer='layer1', style_layers=['layer1', 'layer2'], is_relu=True, avg_pooling=True):
        """
        Initialize the CustomResNet model.

        """
        super(CustomResNet, self).__init__()
        self.resnet18 = models.resnet101(models.ResNet101_Weights.IMAGENET1K_V1)
        self.content_layer = content_layer
        self.style_layers = style_layers
        self.avg_pooling = avg_pooling

        # Replace max pooling with average pooling if avg_pooling is True
        if avg_pooling:
            for name, module in self.resnet18.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    self.resnet18.maxpool = torch.nn.AvgPool2d(kernel_size=3, stride=2,
                                                                  padding=1, ceil_mode=False)

        # Set requires_grad to False for all parameters in the ResNet model
        self.set_requires_grad(False)

    def set_requires_grad(self, requires_grad):
        """
        Set the requires_grad attribute for all parameters in the ResNet model.

        Parameters:
        - requires_grad (bool): If True, parameters will require gradients. If False, they won't.

        """
        for param in self.resnet18.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, is_style=False, is_generated=False):
        """
        Forward pass of the CustomResNet model.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - is_style (bool, optional): If True, extract style features. Default is False.
        - is_generated (bool, optional): If True, return both content and style features. Default is False.

        Returns:
        - content_features (torch.Tensor): Content features if is_generated is False.
        - style_features (list of torch.Tensor): Style features if is_style or is_generated is True.

        """
        style_features = []
        for name, layer in self.resnet18._modules.items():
            if name == 'avgpool':
                break
            x = layer(x)
            if (is_style or is_generated) and name in self.style_layers:
                style_features.append(x)
            if name == self.content_layer:
                content_features = x

        if is_generated:
            return content_features, style_features
        if is_style:
            return style_features
        return content_features
