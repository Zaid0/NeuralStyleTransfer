import torch
import torchvision.models as models
import torch.nn as nn

class VGG19(nn.Module):
    """
    Customized VGG19 model for neural style transfer.

    Parameters:
    - content_layer (str, optional): Name of the content layer. Default is 'conv4_1'.
    - style_layers (list, optional): List of names of style layers. Default is ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'].
    - is_relu (bool, optional): If True, uses ReLU activation after each layer. Default is False.
    - avg_pooling (bool, optional): If True, replaces max pooling with average pooling. Default is True.

    Example:
    >>> model = VGG19(content_layer='conv3_1', style_layers=['conv1_1', 'conv4_1'], is_relu=True)
    >>> content_features, style_features = model(input_tensor, is_generated=True)
    """

    def __init__(self, content_layer='conv4_1', style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                 is_relu=False, avg_pooling=True):
        """
        Initialize the VGG19 model.

        """
        super(VGG19, self).__init__()

        self.vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.vgg19 = nn.Sequential(*[self.vgg19.features[i] for i in range(32)])
        self.content_layer = content_layer
        self.style_layers = style_layers
        self.is_relu = is_relu
        self.ln_to_index_conv = {
            'conv1_1': '0',
            'conv2_1': '5',
            'conv3_1': '10',
            'conv3_2': '12',
            'conv4_1': '19',
            'conv4_2': '21',
            'conv5_1': '28'
        }

        self.ln_to_index_relu = {
            'conv1_1': '1',
            'conv2_1': '6',
            'conv3_1': '11',
            'conv3_2': '13',
            'conv4_1': '20',
            'conv4_2': '22',
            'conv5_1': '29'
        }
        if avg_pooling:
            for name, module in self.vgg19.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    self.vgg19[int(name)] = torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.set_requires_grad(False)

    def set_requires_grad(self, requires_grad):
        """
        Set the requires_grad attribute for all parameters in the VGG19 model.

        Parameters:
        - requires_grad (bool): If True, parameters will require gradients. If False, they won't.

        """
        for param in self.vgg19.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, is_style=False, is_generated=False):
        """
        Forward pass of the VGG19 model.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - is_style (bool, optional): If True, extract style features. Default is False.
        - is_generated (bool, optional): If True, return both content and style features. Default is False.

        Returns:
        - content_features (torch.Tensor): Content features if is_generated is False.
        - style_features (list of torch.Tensor): Style features if is_style or is_generated is True.

        """
        ln_to_index = self.ln_to_index_relu if self.is_relu else self.ln_to_index_conv
        sl = [ln_to_index[i] for i in self.style_layers]
        style_features = []
        for name, layer in self.vgg19._modules.items():
            x = layer(x)

            if (is_style or is_generated) and name in sl:
                    style_features.append(x)

            if name == ln_to_index[self.content_layer]:
                content_features = x

        if is_generated:
            return content_features, style_features
        if is_style:
            return style_features
        return content_features
