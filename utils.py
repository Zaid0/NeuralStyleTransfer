from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import io

def load_imgs(content_img_path, style_img_path, shape, device, white_noise=False, app=False):
    """
    Load and preprocess content and style images for neural style transfer.

    Parameters:
    - content_img_path (str): File path of the content image.
    - style_img_path (str): File path of the style image.
    - shape (int or tuple): Target size of the images after resizing.
                           If int, images will be resized to (shape, shape).
                           If tuple, images will be resized to the specified dimensions.
    - device (torch.device): The device (CPU or GPU) on which to load the images.
    - white_noise (bool, optional): If True, use white noise as the initial generated image.
                                   Default is False.
    - app (bool, optional): If True, assume images are already preprocessed.
                           Default is False.

    Returns:
    - content_img (torch.Tensor): Preprocessed content image.
    - style_img (torch.Tensor): Preprocessed style image.
    - generated_img (torch.Tensor): Initial generated image for neural style transfer.
    - content_name (str, optional): Name of the content image file (if not 'app' mode).
    - style_name (str, optional): Name of the style image file (if not 'app' mode).

    Example:
    >>> content, style, generated = load_imgs("content.jpg", "style.jpg", shape=256, device=device)
    """

    if app:
        content_img = load_img(content_img_path, size=shape, app=app).to(device)
        style_img = load_img(style_img_path, size=shape, app=app).to(device)
        if white_noise:
            generated_img = generate_white_noise_image(shape).to(device).requires_grad_(True)
        else:
            generated_img = content_img.clone().requires_grad_(True).to(device)
        return content_img, style_img, generated_img

    content_name = content_img_path.split('/')[1].split('.')[0]
    style_name = style_img_path.split('/')[1].split('.')[0]
    content_img = load_img(content_img_path, size=shape).to(device)
    style_img = load_img(style_img_path, size=shape).to(device)
    if white_noise:
        generated_img = generate_white_noise_image(shape).to(device).requires_grad_(True)
    else:
        generated_img = content_img.clone().requires_grad_(True).to(device)
    return content_img, style_img, generated_img, content_name, style_name



def load_img(img, size, app=False):
    """
    Load and preprocess an image from the specified path.

    Parameters:
    - img (str): The file path of the image.
    - size (int or tuple): The target size of the image after resizing.
                          If int, the image will be resized to (size, size).
                          If tuple, the image will be resized to the specified dimensions.
    - app (bool, optional): If True, the image is assumed to be already preprocessed,
                            and no further transformation is applied. Default is False.

    Returns:
    - img (torch.Tensor): A preprocessed PyTorch tensor representing the image.
                         The tensor has dimensions (1, C, H, W), where C is the number of channels.

    Example:
    >>> img = load_img("path/to/image.jpg", size=(256,256))
    """

    # If app is False, open the image and convert it to RGB format
    if not app:
        img = Image.open(img).convert("RGB")

    # Define a series of image transformations using torchvision.transforms.Compose
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.5, 0.5, 0.5)),
                std=torch.tensor((0.5, 0.5, 0.5))
            ),
        ]
    )

    # Apply the transformations to the image and add a batch dimension
    img = transform(img).unsqueeze(0)

    return img


def generate_white_noise_image(shape=(512, 682), normalize=True):
    """
    Generate a white noise image in RGB format with the specified shape using PyTorch.

    Parameters:
    - shape: Tuple (height, width) specifying the shape of the image.

    Returns:
    - A PyTorch tensor representing the white noise image in RGB format.
    """
    noise = torch.randn(1, 3, shape[0], shape[1], dtype=torch.float32)
    if normalize:
        noise = (noise+1)/2
    return noise


def imcnvt(image):
    """
    Convert a PyTorch tensor representing an image to a NumPy array and apply post-processing.

    Parameters:
    - image (torch.Tensor): A PyTorch tensor representing the image.

    Returns:
    - numpy.ndarray: A NumPy array representing the post-processed image.

    Example:
    >>> img_tensor = load_img("path/to/image.jpg", size=(256, 256))
    >>> img_numpy = imcnvt(img_tensor)
    """

    # Convert the image tensor to a NumPy array
    x = image.to("cpu").clone().detach().numpy().squeeze()

    # Transpose the dimensions to (height, width, channels)
    x = x.transpose(1, 2, 0)

    # Rescale the values to the original image range
    x = x * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

    # Clip the values to be in the range [0, 1]
    return np.clip(x, 0, 1)


def display_image(image, title ='Epoch ', i=None):
    """
    Display the generated image using Matplotlib.

    Parameters:
    - image: The image (PyTorch tensor) to display.
    - title: Title for the displayed image.
    """
    t = title+str(i) if title == 'Epoch ' else title

    plt.imshow(imcnvt(image), label=t)
    plt.show()

