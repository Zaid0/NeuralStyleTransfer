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


class InceptionV3(nn.Module):
    """
    Customized InceptionV3 model for neural style transfer.

    Parameters:
    - content_layer (str, optional): Name of the content layer. Default is 'Mixed_5b.branch3x3dbl_2.conv'.
    - style_layers (list, optional): List of names of style layers. Default is ['Mixed_5b.branch1x1.conv',
                                 'Mixed_5b.branch5x5_1.conv',
                                 'Mixed_5b.branch5x5_2.conv',
                                'Mixed_5b.branch3x3dbl_1.conv'].

    Example:
    model = InceptionV3(content_layer='Mixed_6a.branch3x3.conv', style_layers=['Mixed_5b.branch1x1.conv'])
     content_features, style_features = model(input_tensor, is_generated=True)
    """

    def __init__(self, content_layer='Mixed_5b.branch3x3dbl_2.conv', style_layers=['Mixed_5b.branch1x1.conv',
                                 'Mixed_5b.branch5x5_1.conv',
                                 'Mixed_5b.branch5x5_2.conv',
                                'Mixed_5b.branch3x3dbl_1.conv']):
        """
        Initialize the InceptionV3 model.

        """
        super(InceptionV3, self).__init__()
        self.inception = models.inception_v3(models.Inception_V3_Weights.IMAGENET1K_V1)
        self.content_layer = content_layer
        self.style_layers = style_layers

        # Set requires_grad to False for all parameters in the InceptionV3 model
        self.set_requires_grad(False)

    def set_requires_grad(self, requires_grad):
        """
        Set the requires_grad attribute for all parameters in the InceptionV3 model.

        Parameters:
        - requires_grad (bool): If True, parameters will require gradients. If False, they won't.

        """
        for param in self.inception.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, is_style=False, is_generated=False):
        """
        Forward pass of the InceptionV3 model.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - is_style (bool, optional): If True, extract style features. Default is False.
        - is_generated (bool, optional): If True, return both content and style features. Default is False.

        Returns:
        - content_features (torch.Tensor): Content features if is_generated is False.
        - style_features (list of torch.Tensor): Style features if is_style or is_generated is True.

        """
        style_features = []
        for name, layer in self.inception._modules.items():
            if name == 'AuxLogits':
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



# ['Mixed_5b.branch1x1.conv', 'Mixed_5b.branch5x5_1.conv', 'Mixed_5b.branch5x5_2.conv',
#  'Mixed_5b.branch3x3dbl_1.conv', 'Mixed_5b.branch3x3dbl_2.conv', 'Mixed_5b.branch3x3dbl_3.conv',
#  'Mixed_5b.branch_pool.conv', 'Mixed_5c.branch1x1.conv', 'Mixed_5c.branch5x5_1.conv',
#  'Mixed_5c.branch5x5_2.conv', 'Mixed_5c.branch3x3dbl_1.conv', 'Mixed_5c.branch3x3dbl_2.conv',
#  'Mixed_5c.branch3x3dbl_3.conv', 'Mixed_5c.branch_pool.conv', 'Mixed_5d.branch1x1.conv',
#  'Mixed_5d.branch5x5_1.conv', 'Mixed_5d.branch5x5_2.conv', 'Mixed_5d.branch3x3dbl_1.conv',
#  'Mixed_5d.branch3x3dbl_2.conv', 'Mixed_5d.branch3x3dbl_3.conv', 'Mixed_5d.branch_pool.conv',
#  'Mixed_6a.branch3x3.conv', 'Mixed_6a.branch3x3dbl_1.conv', 'Mixed_6a.branch3x3dbl_2.conv',
#  'Mixed_6a.branch3x3dbl_3.conv', 'Mixed_6b.branch1x1.conv', 'Mixed_6b.branch7x7_1.conv',
#  'Mixed_6b.branch7x7_2.conv', 'Mixed_6b.branch7x7_3.conv', 'Mixed_6b.branch7x7dbl_1.conv',
#  'Mixed_6b.branch7x7dbl_2.conv', 'Mixed_6b.branch7x7dbl_3.conv', 'Mixed_6b.branch7x7dbl_4.conv',
#  'Mixed_6b.branch7x7dbl_5.conv', 'Mixed_6b.branch_pool.conv', 'Mixed_6c.branch1x1.conv',
#  'Mixed_6c.branch7x7_1.conv', 'Mixed_6c.branch7x7_2.conv', 'Mixed_6c.branch7x7_3.conv',
#  'Mixed_6c.branch7x7dbl_1.conv', 'Mixed_6c.branch7x7dbl_2.conv', 'Mixed_6c.branch7x7dbl_3.conv',
#  'Mixed_6c.branch7x7dbl_4.conv', 'Mixed_6c.branch7x7dbl_5.conv', 'Mixed_6c.branch_pool.conv',
#  'Mixed_6d.branch1x1.conv', 'Mixed_6d.branch7x7_1.conv', 'Mixed_6d.branch7x7_2.conv',
#  'Mixed_6d.branch7x7_3.conv', 'Mixed_6d.branch7x7dbl_1.conv', 'Mixed_6d.branch7x7dbl_2.conv',
#  'Mixed_6d.branch7x7dbl_3.conv', 'Mixed_6d.branch7x7dbl_4.conv', 'Mixed_6d.branch7x7dbl_5.conv',
#  'Mixed_6d.branch_pool.conv', 'Mixed_6e.branch1x1.conv', 'Mixed_6e.branch7x7_1.conv',
#  'Mixed_6e.branch7x7_2.conv', 'Mixed_6e.branch7x7_3.conv', 'Mixed_6e.branch7x7dbl_1.conv',
#  'Mixed_6e.branch7x7dbl_2.conv', 'Mixed_6e.branch7x7dbl_3.conv', 'Mixed_6e.branch7x7dbl_4.conv',
#  'Mixed_6e.branch7x7dbl_5.conv', 'Mixed_6e.branch_pool.conv', 'Mixed_7a.branch3x3_1.conv',
#  'Mixed_7a.branch3x3_2.conv', 'Mixed_7a.branch7x7x3_1.conv',
#  'Mixed_7a.branch7x7x3_2.conv', 'Mixed_7a.branch7x7x3_3.conv', 'Mixed_7a.branch7x7x3_4.conv',
#  'Mixed_7b.branch1x1.conv', 'Mixed_7b.branch3x3_1.conv', 'Mixed_7b.branch3x3_2a.conv',
#  'Mixed_7b.branch3x3_2b.conv', 'Mixed_7b.branch3x3dbl_1.conv', 'Mixed_7b.branch3x3dbl_2.conv',
#  'Mixed_7b.branch3x3dbl_3a.conv', 'Mixed_7b.branch3x3dbl_3b.conv', 'Mixed_7b.branch_pool.conv',
#  'Mixed_7c.branch1x1.conv', 'Mixed_7c.branch3x3_1.conv', 'Mixed_7c.branch3x3_2a.conv',
#  'Mixed_7c.branch3x3_2b.conv', 'Mixed_7c.branch3x3dbl_1.conv', 'Mixed_7c.branch3x3dbl_2.conv',
#  'Mixed_7c.branch3x3dbl_3a.conv', 'Mixed_7c.branch3x3dbl_3b.conv', 'Mixed_7c.branch_pool.conv']











