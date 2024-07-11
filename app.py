import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Print versions for debugging
st.write("TensorFlow version:", tf.__version__)

# model 
out_len = 10  # Replace this with the actual number of output classes

# Ensure no conflicts with 'model' or 'load_model'
model_path = 'tomato_model'

# Load your pre-trained model
try:
    VIT = load_model(model_path)
    st.write("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the class names
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
]

# Function to load and preprocess the image
def load_and_prep_image(image):
    try:
        img = image.resize((224, 224))  # Assuming your model expects 224x224 images
        img = np.array(img) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit app
st.title("Tomato Disease Detection")
st.write("Upload an image of a tomato leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        prepped_image = load_and_prep_image(image)
        
        # Ensure the model is loaded and image is preprocessed before making a prediction
        if VIT is not None and prepped_image is not None:
            # Make prediction
            prediction = VIT.predict(prepped_image)
            predicted_class = class_names[np.argmax(prediction)]
            
            # Display the prediction
            st.write(f"Prediction: {predicted_class}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
