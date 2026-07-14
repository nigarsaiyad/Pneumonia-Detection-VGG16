import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide"
)


# =====================================================
# Custom CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f4f8fb;
    }


    .header {

        background: linear-gradient(
            90deg,
            #003973,
            #005aa7
        );

        padding:25px;
        border-radius:15px;
        color:white;
        text-align:center;

    }


    .card {

        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 5px 15px rgba(0,0,0,0.1);

    }


    </style>

    """,
    unsafe_allow_html=True
)



# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "vgg16_pneumonia.keras"
    )

    return model


model = load_model()



# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("🫁 Pneumonia AI")


    st.write(
        """
        ### Model Details

        **Architecture**
        
        VGG16 Transfer Learning


        **Input Size**

        224 × 224 RGB


        **Classes**

        🟢 NORMAL

        🔴 PNEUMONIA


        **Framework**

        TensorFlow + Streamlit

        """
    )



# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div class="header">

    <h1>
    🫁 AI Chest X-Ray Diagnosis System
    </h1>

    <h3>
    Pneumonia Detection using VGG16 Deep Learning
    </h3>

    </div>

    """,
    unsafe_allow_html=True
)


st.write("")



# =====================================================
# Image Preprocessing
# SAME AS TRAINING
# =====================================================

def preprocess_image(image):

    # Resize image
    image = image.resize(
        (224,224)
    )


    # Convert grayscale image to RGB
    image = image.convert(
        "RGB"
    )


    # Convert to numpy array
    image = np.array(
        image
    )


    # Normalize pixel values
    image = image / 255.0


    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )


    return image



# =====================================================
# Upload Image
# =====================================================

uploaded_file = st.file_uploader(

    "📤 Upload Chest X-Ray Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



# =====================================================
# Prediction
# =====================================================

if uploaded_file:


    image = Image.open(
        uploaded_file
    )


    col1, col2 = st.columns(2)



    with col1:

        st.subheader(
            "📷 Uploaded X-Ray"
        )


        st.image(
            image,
            use_container_width=True
        )



    processed_image = preprocess_image(
        image
    )


    prediction = model.predict(
        processed_image
    )


    probability = prediction[0][0]



    with col2:

        st.subheader(
            "🤖 AI Prediction"
        )


        if probability >= 0.5:


            result = "PNEUMONIA"

            confidence = probability


            st.error(
                "🔴 PNEUMONIA DETECTED"
            )


        else:


            result = "NORMAL"

            confidence = 1 - probability


            st.success(
                "🟢 NORMAL"
            )



        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )


        st.progress(
            float(confidence)
        )



        st.write(
            "Raw Model Output:"
        )

        st.write(
            prediction
        )



# =====================================================
# Model Performance
# =====================================================

st.divider()


st.subheader(
    "📊 Model Performance"
)


a,b,c,d = st.columns(4)


a.metric(
    "Accuracy",
    "86.7%"
)


b.metric(
    "Precision",
    "86.6%"
)


c.metric(
    "Recall",
    "93.1%"
)


d.metric(
    "AUC",
    "93.6%"
)



# =====================================================
# About
# =====================================================

st.divider()


st.subheader(
    "🧠 About This Project"
)


st.write(
    """
    This application uses a VGG16 Convolutional Neural Network
    with Transfer Learning to classify chest X-ray images.

    The model was trained to detect:

    - NORMAL lungs
    - Pneumonia infected lungs


    Technologies used:

    • Python

    • TensorFlow/Keras

    • VGG16 CNN

    • Streamlit

    """
)



# =====================================================
# Disclaimer
# =====================================================

st.warning(
    """
    ⚠️ This application is created for educational purposes only.
    It should not be considered a medical diagnosis system.
    """
)