import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pre-trained model
# model = load_model('tomato_disease_model.h5')

# Define the class names
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
]

# Function to load and preprocess the image
def load_and_prep_image(image):
    img = image.resize((224, 224))  # Assuming your model expects 224x224 images
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
st.title("Tomato Disease Detection")
st.write("Upload an image of a tomato leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    prepped_image = load_and_prep_image(image)
    
    # Make prediction
    # prediction = model.predict(prepped_image)
    # predicted_class = class_names[np.argmax(prediction)]
    predicted_class = class_names[np.argmax([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
    
    # Display the prediction
    st.write(f"Prediction: {predicted_class}")
