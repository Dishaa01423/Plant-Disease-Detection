import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Tomato Disease Detective", page_icon="🍅", layout="wide")

# Custom CSS to inject into the Streamlit app
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
    color: #FF6347;
}
.medium-font {
    font-size:20px !important;
    font-weight: bold;
    color: #228B22;
}
.stButton>button {
    color: #fff;
    background-color: #FF6347;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown('<p class="big-font">🍅 Tomato Disease Detective 🕵️‍♂️</p>', unsafe_allow_html=True)

# Sidebar for additional information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg", use_column_width=True)
    st.markdown("## About")
    st.info("This app uses AI to detect diseases in tomato plants. Simply upload an image of a tomato leaf, and let the detective do its work!")
    st.markdown("## How to use")
    st.write("1. Upload a clear image of a tomato leaf")
    st.write("2. Wait for the AI to analyze")
    st.write("3. Review the diagnosis and recommendations")
    st.markdown("## 🧑‍🌾 Happy Gardening! 🌱")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<p class="medium-font">Upload Your Tomato Leaf Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model and other setup (as in your original code)
model_path = 'tomato_model'
try:
    VIT = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the class names
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Define cure and prevention suggestions for each disease
disease_info = {
    'Tomato_Bacterial_spot': {
        'cure': [
            "Remove infected plants",
            "Apply copper-based fungicides",
            "Prune to improve air circulation"
        ],
        'prevention': [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Avoid overhead irrigation",
            "Disinfect gardening tools regularly"
        ]
    },
    'Tomato_Early_blight': {
        'cure': [
            "Remove infected leaves",
            "Apply fungicides (chlorothalonil, mancozeb, or copper)",
            "Improve air circulation around plants"
        ],
        'prevention': [
            "Mulch around plants",
            "Ensure good air circulation",
            "Water at the base of plants",
            "Rotate crops every 2-3 years"
        ]
    },
    'Tomato_Late_blight': {
        'cure': [
            "Remove and destroy infected plants",
            "Apply fungicides (chlorothalonil or copper-based)",
            "Harvest remaining healthy fruit"
        ],
        'prevention': [
            "Plant resistant varieties",
            "Avoid overhead watering",
            "Space plants properly",
            "Remove volunteers and nightshade weeds"
        ]
    },
    'Tomato_Leaf_Mold': {
        'cure': [
            "Improve air circulation",
            "Apply fungicides (chlorothalonil or mancozeb)",
            "Remove severely infected leaves"
        ],
        'prevention': [
            "Reduce humidity in greenhouses",
            "Avoid leaf wetness",
            "Use resistant varieties",
            "Prune and stake plants for better air flow"
        ]
    },
    'Tomato_Septoria_leaf_spot': {
        'cure': [
            "Remove infected leaves",
            "Apply fungicides (chlorothalonil or copper-based)",
            "Improve air circulation"
        ],
        'prevention': [
            "Mulch around plants",
            "Practice crop rotation",
            "Avoid overhead watering",
            "Remove plant debris after harvest"
        ]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'cure': [
            "Use insecticidal soaps or neem oil",
            "Introduce predatory mites",
            "Prune heavily infested leaves"
        ],
        'prevention': [
            "Keep plants well-watered",
            "Increase humidity",
            "Use reflective mulches",
            "Avoid excessive nitrogen fertilization"
        ]
    },
    'Tomato_Target_Spot': {
        'cure': [
            "Remove infected leaves",
            "Apply fungicides (chlorothalonil or copper-based)",
            "Improve air circulation"
        ],
        'prevention': [
            "Improve air circulation",
            "Avoid overhead watering",
            "Practice crop rotation",
            "Remove plant debris after harvest"
        ]
    },
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': {
        'cure': [
            "No cure available",
            "Remove and destroy infected plants",
            "Control whitefly population"
        ],
        'prevention': [
            "Use resistant varieties",
            "Control whiteflies with insecticides or traps",
            "Use reflective mulches",
            "Plant early in the season"
        ]
    },
    'Tomato_Tomato_mosaic_virus': {
        'cure': [
            "No cure available",
            "Remove and destroy infected plants",
            "Control aphid population"
        ],
        'prevention': [
            "Use disease-free seeds",
            "Disinfect tools and hands",
            "Control aphids",
            "Avoid working with plants when wet"
        ]
    },
    'Tomato_healthy': {
        'cure': [
            "No treatment needed"
        ],
        'prevention': [
            "Maintain good soil health",
            "Water consistently",
            "Provide adequate sunlight",
            "Monitor for early signs of pests or diseases"
        ]
    }
}

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

# Prediction and results display
if uploaded_file is not None:
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.markdown('<p class="medium-font">Analysis Results</p>', unsafe_allow_html=True)
        
        with st.spinner('Analyzing image... 🔍'):
            # Preprocess the image
            prepped_image = load_and_prep_image(image)

            # Ensure the model is loaded and image is preprocessed before making a prediction
            if VIT is not None and prepped_image is not None:
                # Make prediction
                prediction = VIT.predict(prepped_image)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                # Display the prediction with a progress bar
                st.write(f"Detected Disease: **{predicted_class}**")
                st.progress(confidence / 100)
                st.write(f"Confidence: {confidence:.2f}%")

                # Create a pie chart for top 3 predictions
                top_3_idx = np.argsort(prediction[0])[-3:][::-1]
                top_3_values = prediction[0][top_3_idx] * 100
                top_3_labels = [class_names[i] for i in top_3_idx]

                fig = go.Figure(data=[go.Pie(labels=top_3_labels, values=top_3_values, hole=.3)])
                fig.update_layout(title_text="Top 3 Predictions")
                st.plotly_chart(fig)

                # Display cure and prevention suggestions
                if predicted_class in disease_info:
                    st.markdown("### 🏥 Cure:")
                    for cure_point in disease_info[predicted_class]['cure']:
                        st.markdown(f"- {cure_point}")
                    
                    st.markdown("### 🛡️ Prevention:")
                    for prevention_point in disease_info[predicted_class]['prevention']:
                        st.markdown(f"- {prevention_point}")
                else:
                    st.write("No specific cure or prevention information available for this condition.")

        # Add a "Get Expert Help" button
        if st.button("Get Expert Help"):
            st.markdown("### 👩‍🌾 Expert Consultation")
            st.write("Our team of agricultural experts is ready to assist you. Please provide the following information:")
            with st.form("expert_help_form"):
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
                description = st.text_area("Describe your problem in detail")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.success("Thank you! Our experts will contact you within 24 hours.")

    # Add a fun fact about tomatoes
    st.markdown("---")
    st.markdown("### 🍅 Fun Tomato Fact")
    fun_facts = [
        "Tomatoes are actually fruits, not vegetables!",
        "The fear of tomatoes is called Lycopersicophobia.",
        "The world's largest tomato tree produces more than 32,000 tomatoes a year!",
        "There are over 10,000 varieties of tomatoes worldwide.",
        "The first tomatoes discovered by Europeans were yellow, hence the Italian name 'pomodoro' (golden apple)."
    ]
    st.write(np.random.choice(fun_facts))

else:
    st.write("Please upload an image to get started!")

# Footer
st.markdown("---")
