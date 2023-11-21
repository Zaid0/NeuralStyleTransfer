from train import train


content_img_path = "contents/seasky.jpg"
style_img_path = "styles/starry_night 1.jpeg"
shape = (512, 682)
total_steps = 4000
lr = .007
alpha = 1.
beta = 1000000.


# vgg19 layers:
# 'conv1_1', 'conv2_1', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1'

# Inception layers:
# Conv2d_1a_3x3
# Conv2d_2a_3x3
# Conv2d_2b_3x3
# maxpool1
# Conv2d_3b_1x1
# Conv2d_4a_3x3
# maxpool2
# Mixed_5b
# Mixed_5c
# Mixed_5d
# Mixed_6a
# Mixed_6b
# Mixed_6c
# Mixed_6d
# Mixed_6e

# resnet layers:
# conv1
# bn1
# relu
# maxpool
# layer1
# layer2
# layer3
# layer4

#
# Conv2d_1a_3x3
# Conv2d_2a_3x3
# Conv2d_2b_3x3
# maxpool1
# Conv2d_3b_1x1
# Conv2d_4a_3x3
# maxpool2
# Mixed_5b
# Mixed_5c
# Mixed_5d
# Mixed_6a
# Mixed_6b
# Mixed_6c
# Mixed_6d
# Mixed_6e


train(
    content_img_path=content_img_path,
    style_img_path=style_img_path,
    model_name='VGG19',
    shape=shape,
    alpha=alpha,
    beta=beta,
    lr=lr,
    total_steps=total_steps,
    avg_pooling=True,
    content_layer='conv3_2',
    style_layers= ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'],
    white_noise=False,
    app=False
)
