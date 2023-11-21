from pathlib import Path
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import imcnvt, load_imgs
from model import VGG19
from resnet import CustomResNet
from NSTLoss import NSTLoss
import torch
from inception import InceptionV3



def train(
        content_img_path,
        style_img_path,
        shape=(512, 512),
        model_name='vgg19',
        path='generated/',
        alpha=1.,
        beta=1000.,
        wl=[1, 1, 1, 1, 1],
        lr=0.07,
        log_save=100,
        log_loss=10,
        content_layer='conv3_2',
        style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'],
        total_steps=1000,
        optimizer='Adam',
        is_relu=True,
        avg_pooling=True,
        white_noise=False,
        app=False
):
    """
    Train a neural style transfer model.

    Parameters:
    - content_img_path (str): File path of the content image.
    - style_img_path (str): File path of the style image.
    - shape (tuple): Target size of the images after resizing.
    - model_name (str): Name of the neural style transfer model ('vgg19', 'resnet101', 'inception_v3').
    - path (str, optional): Directory path to save the generated images. Default is 'generated/'.
    - alpha (float, optional): Weight for content loss. Default is 1.0.
    - beta (float, optional): Weight for style loss. Default is 1000.0.
    - wl (list, optional): Weight for each style layer. Default is [1, 1, 1, 1, 1].
    - lr (float, optional): Learning rate for the optimizer. Default is 0.07.
    - log_save (int, optional): Frequency of saving generated images. Default is 100.
    - log_loss (int, optional): Frequency of logging the loss. Default is 10.
    - content_layer (str, optional): Name of the content layer. Default is 'conv3_2'.
    - style_layers (list, optional): List of names of style layers. Default is ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'].
    - total_steps (int, optional): Total training steps. Default is 1000.
    - optimizer (str, optional): Optimizer for training ('Adam' is the only supported option). Default is 'Adam'.
    - is_relu (bool, optional): If True, uses ReLU activation in the neural style transfer model. Default is True.
    - avg_pooling (bool, optional): If True, replaces max pooling with average pooling in the neural style transfer model. Default is True.
    - white_noise (bool, optional): If True, use white noise as the initial generated image. Default is False.
    - app (bool, optional): If True, assume images are already preprocessed. Default is False.

    Returns:
    - torch.Tensor: The generated image tensor.

    Example:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # Load or preprocess images based on the 'app' flag
    if not app:
        content_img, style_img, generated_img, content_name, style_name = load_imgs(
            content_img_path,
            style_img_path,
            shape,
            device,
            white_noise=white_noise
        )
        directory_path = Path(path)
        folder_name = f"{model_name}_{content_name}_{style_name}_{shape}_beta_{beta}_alpha_{alpha}_lr_{lr}_{content_layer}_{style_layers[-1]}_{is_relu}_white_noise_{white_noise}"

        new_folder_path = directory_path / folder_name
        if not new_folder_path.exists():
            new_folder_path.mkdir()
        else:
            print("Folder already exists")
            user_input = input("Would you like to continue (y/n): ")
            if user_input.lower() == "n":
                exit()

    else:
        content_img, style_img, generated_img = load_imgs(
            content_img_path,
            style_img_path,
            shape,
            device,
            white_noise=white_noise,
            app=True
        )

    # Initialize NSTLoss and optimizer
    nst_loss = NSTLoss(alpha=alpha, beta=beta, wl=wl)
    if optimizer == 'Adam':
        optimizer = optim.Adam([generated_img], lr)
    else:
        print("Optimizer is not compatible")
        exit()

    # Initialize the selected neural style transfer model
    if model_name == 'VGG19':
        model = VGG19(content_layer=content_layer, style_layers=style_layers,
                      is_relu=is_relu, avg_pooling=avg_pooling).to(device)
    elif model_name == 'ResNet':
        model = CustomResNet(
            content_layer=content_layer,
            style_layers=style_layers,
            avg_pooling=avg_pooling
        ).to(device)
    elif model_name == 'Inception':
        model = InceptionV3(content_layer=content_layer,
                            style_layers=style_layers).to(device)
    else:
        print('Model is not supported')
        exit()

    model.eval()

    # Extract features from content and style images
    with torch.no_grad():
        content_features = model(content_img, is_style=False, is_generated=False)
        style_features = model(style_img, is_style=True, is_generated=False)

    # Training loop
    for i in range(total_steps + 1):
        generated_content_features, generated_style_features = model(generated_img, is_generated=True)
        loss = nst_loss(content_features, style_features, generated_content_features, generated_style_features)

        # Log loss at specified intervals
        if i % log_loss == 0:
            print("Epoch ", i, " ", loss)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save generated images at specified intervals
        if i % log_save == 0 and not app:
            plt.imsave(new_folder_path / f'generated_{i}.png', imcnvt(generated_img), format='png')

    return generated_img
