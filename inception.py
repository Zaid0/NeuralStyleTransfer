import torch.nn as nn
from torchvision import models

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











