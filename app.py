import streamlit as st
from PIL import Image
from train import train  # Import your train function from the train.py file
from utils import imcnvt
import os
import base64


# todo:
# 5. decrese size of generated img

st.set_page_config(layout='wide')
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

vgg19_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1']
inception_layers = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool2', 'Mixed_5b',
                    'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e']
resnet_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

def home_page(session_state):
    st.title("Artistic Style Transfer Project :art:")

    st.write("""
    Welcome to the Artistic Style Transfer project! This web application allows you to unleash your creativity 
    by transforming your images into unique pieces of art. Through the magic of Neural Style Transfer (NST), 
    you can apply the style of iconic paintings to your photos and witness the fascinating fusion of content 
    and artistic flair. Experiment with different neural network models to achieve diverse and captivating results.
    """)

    st.header("Project Overview :books:")

    st.write("""
    Neural Style Transfer (NST) is a powerful deep learning technique that marries the content of one image with 
    the artistic style of another. In this project, a sophisticated neural network is trained to discern patterns, 
    textures, and colors, enabling it to seamlessly blend the distinctive style of a chosen reference image with 
    the content of your original image. Immerse yourself in the world of digital artistry and discover the 
    captivating transformations you can achieve with just a few clicks!
    """)

    # Add links or buttons for examples
    st.subheader("Examples:")
    example('Example 1: Transforming A Dog Image into digital style',  "assets/contents/dog.jpg", "assets/styles/digital_art_1.jpg", "assets/outputs/dog_ouput.png", flag=False,
            )

    example('Example 2: Transforming a sea Image into Starry Night Style', "assets/contents/seasky.jpg", "assets/styles/starry_night 1.jpeg", "assets/outputs/seasky_ouput.png",
            flag=True)

    # Add a button to try neural style transfer

    try_neural_style_transfer(session_state)

def example(style_description, content_path, style_path, output_path, flag=False):

    # Use st.markdown with a custom CSS style to adjust the margin
    if not flag:
        st.markdown(
            f"<h3 style='margin-bottom: 20px;'>{style_description}</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h3 style='margin-bottom: 20px;'>{style_description}</h3>",
            unsafe_allow_html=True
        )

    col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 3])

    st.markdown(
        """
        <style>
            .center {
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .sign {
                font-size: 15em;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with col1:
        st.image(content_path, caption="Content Image", use_column_width=True)

    with col2:
        st.markdown(
            """
            <div class="center">
                <div class="sign">+</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display style image with decreased margin
    with col3:
        st.image(style_path, caption="Style Image", use_column_width=True)

    with col4:
        st.markdown(
            """
            <div class="center">
                <div class="sign">=</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display generated output image with decreased margin
    with col5:
        st.image(output_path, caption=f"Generated Image", use_column_width=True)

    st.markdown(
        """
        <style>
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]) {
                align-items: center;
            }
        </style>
        """,
        unsafe_allow_html=True)



def try_neural_style_transfer(session_state):

    vgg19_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1']

    inception_layers = [
        'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1',
        'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool2', 'Mixed_5b',
        'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b',
        'Mixed_6c', 'Mixed_6d', 'Mixed_6e'
    ]

    resnet_layers = [
        'conv1', 'bn1', 'relu', 'maxpool',
        'layer1', 'layer2', 'layer3', 'layer4'
    ]

    # Streamlit app title and description
    st.title("Neural Style Transfer App")
    st.write("Transform your images with artistic styles using Neural Style Transfer.")

    contents_path = 'assets/contents'
    contents_image_paths = [file for file in os.listdir(contents_path)]

    style_path = 'assets/styles'
    styles_image_paths = [file for file in os.listdir(style_path)]

    # st.sidebar.markdown( f"""
    # <style>
    #     [data-testid="stSidebar"] {{
    #         background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    #         background-size: cover;
    #         background-position: center center;
    #         background-repeat: no-repeat;
    #     }}
    # </style>
    # """,
    # unsafe_allow_html=True)
    # Sidebar for instructions
    st.sidebar.title("Instructions")
    st.sidebar.write("1. Choose a model from the dropdown.")
    st.sidebar.write("2. Upload a content image and a style image.")
    st.sidebar.write("3. Adjust parameters on the right.")
    st.sidebar.write("4. Click 'Start Style Transfer'.")
    st.sidebar.markdown("---")
    st.sidebar.title("Contacts")
    st.sidebar.write("   - Email: zaidrjoub.jobs@gmail.com")
    st.sidebar.markdown("   - GitHub repo: [Click here](https://github.com/Zaid0/NeuralStyleTransfer)")
    st.sidebar.markdown("   - My LinkedIn: [Click here](https://www.linkedin.com/in/zaid-rjoub-495974217/)")

    # Function to encode a file to a data URI
    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.pdf" ' \
               f'style="text-decoration:none; color: #008CBA;">Download {file_label}</a>'
        return href

    # Download CV Section
    st.sidebar.markdown("---")
    st.sidebar.header("Download CV")
    st.sidebar.markdown(get_binary_file_downloader_html('assets/resume.pdf', 'cv'), unsafe_allow_html=True)

    # User-defined parameters for shape
    st.header("Image Shape")

    # Use a single row with two columns for width and height
    col_width, col_height = st.columns(2)

    with col_width:
        width = st.number_input("Width:", min_value=64, value=512)

    with col_height:
        height = st.number_input("Height:", min_value=64, value=512)

    col1, col2 = st.columns(2)
    # Column 1: Upload Content Image
    with col1:
        st.header("Upload Content Image")

        # User can choose from predefined images or upload their own
        option_content_img = st.selectbox("Choose or Upload Content Image", ["Select an image", "Upload your own"])

        if option_content_img == "Select an image":
            # User selects a predefined image
            content_img = st.selectbox("Choose a predefined content image",
                                       contents_image_paths)
            image1 = Image.open(os.path.join(contents_path, content_img)).resize((width, height))
            st.image(image1, caption="Content Image")
        elif option_content_img == "Upload your own":
            # User uploads their own image
            content_img = st.file_uploader("Choose a content image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
            if content_img is not None:
                image1 = Image.open(content_img).resize((width, height))
                st.image(image1, caption="Content Image")

    # Column 2: Upload Style Image
    with col2:
        st.header("Upload Style Image")

        # User can choose from predefined images or upload their own
        option_style_img = st.selectbox("Choose or Upload Style Image", ["Select an image", "Upload your own"])

        if option_style_img == "Select an image":
            # User selects a predefined image
            style_img = st.selectbox("Choose a predefined style image", styles_image_paths)
            image2 = Image.open(os.path.join(style_path, style_img)).resize((width, height))
            st.image(image2, caption="Style Image")
        elif option_style_img == "Upload your own":
            # User uploads their own image
            style_img = st.file_uploader("Choose a style image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
            if style_img is not None:
                image2 = Image.open(style_img).resize((width, height))
                st.image(image2, caption="Style Image")

    # User-defined parameters in a single column
    st.header("Style Transfer Parameters")
    # Dropdown menu to choose the model
    selected_model = st.selectbox("Choose a Model:", ["VGG19", "Inception", "ResNet"])
    # Define available layers based on the selected model
    if selected_model == "VGG19":
        available_layers = vgg19_layers
    elif selected_model == "Inception":
        available_layers = inception_layers
    elif selected_model == "ResNet":
        available_layers = resnet_layers

    # Adjusting the layout for better organization
    col_params1, col_params2 = st.columns(2)

    with col_params1:
        alpha = st.slider("Content Weight (Alpha):", 0.0, 100.0, 1.0)
        lr = st.slider("Learning Rate:", 0.01, 1.0, 0.01)

    with col_params2:
        beta = st.slider("Style Weight (Beta):", 0.0, 10000.0, 1000.0)
        total_steps = st.slider("Total Steps:", 10, 1000, 50)

    # Use a single row with two columns for content layer and style layers
    col_layers1, col_layers2 = st.columns(2)

    if selected_model == "VGG19":
        default_content_layer = "conv3_2"
        default_style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1"]

    elif selected_model == "Inception":
        default_content_layer = 'Conv2d_1a_3x3'
        default_style_layers = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1']

    elif selected_model == "ResNet":
        default_content_layer = "relu"
        default_style_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']

    # User can select different layers
    with col_layers1:
        content_layer = st.selectbox("Content Layer:", available_layers,
                                     index=available_layers.index(default_content_layer))
        avg_pooling = st.checkbox("Use Average Pooling for Style", value=True)


    with col_layers2:
        style_layers = st.multiselect("Style Layers:", available_layers,
                                      default=[l for l in default_style_layers if l in available_layers])


    # Train the style transfer model
    if st.button("Start Style Transfer", key="start_button"):
        if content_img is not None and style_img is not None:
            # Call your train function here with the provided arguments
            with st.spinner("Applying Style Transfer..."):

                generated_img = train(
                    model_name=selected_model,
                    content_img_path=image1,
                    style_img_path=image2,
                    shape=(height, width),
                    alpha=alpha,
                    beta=beta,
                    lr=lr,
                    total_steps=total_steps,
                    avg_pooling=avg_pooling,
                    content_layer=content_layer,
                    style_layers=style_layers,
                    app=True
                )

            st.title("Generated Image:")
            # Display the generated image
            st.image(imcnvt(generated_img.squeeze(0)), caption="Generated Image", width=width)

        else:
            st.warning("Please upload both content and style images before starting style transfer.")





def main():

    # st.title("Neural Style Transfer App")

    # Streamlit app title and description
    # st.write("Transform your images with artistic styles using Neural Style Transfer.")

    # contents_path = 'contents'
    # contents_image_paths = [file for file in os.listdir(contents_path)]

    # style_path = 'styles'
    # styles_image_paths = [file for file in os.listdir(style_path)]

    css = '''
    <style>
    section.main > div:has (~ footer) {
        padding-bottom: 5px;
    }
    body
    {
    margin: 0 auto;
    }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    session_state = st.session_state
    if 'page' not in session_state:
        session_state.page = 'home'

    if session_state.page == 'home':
        home_page(session_state)
    elif session_state.page == 'neural_style_transfer':
        try_neural_style_transfer(session_state)

    # Call the home_page function to display the home page content

if __name__ == "__main__":
    main()
