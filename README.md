# 🍅 Tomato Disease Detective 🕵️‍♂️

| Key | Value |
|-----|-------|
| title | Plant Disease Detection |
| emoji | 🐠 |
| colorFrom | purple |
| colorTo | gray |
| sdk | streamlit |
| sdk_version | 1.36.0 |
| app_file | app.py |
| pinned | false |

## About

Tomato Disease Detective is an AI-powered web application that helps gardeners and farmers identify diseases in tomato plants. By simply uploading an image of a tomato leaf, users can get instant disease detection results along with cure and prevention suggestions.

## Features

- **Disease Detection**: Utilizes a pre-trained TensorFlow model to identify 10 different tomato plant conditions, including various diseases and healthy plants.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive user experience.
- **Detailed Analysis**: Provides confidence scores, top 3 predictions visualization, and specific cure and prevention tips for each detected disease.
- **Expert Consultation**: Offers a feature to request expert help for more complex cases.
- **Educational Content**: Includes fun facts about tomatoes to engage users.

## Technologies Used

- Python
- Streamlit (sdk version 1.36.0)
- TensorFlow / Keras
- Plotly
- PIL (Python Imaging Library)
- NumPy

## How to Use

1. Clone this repository
2. Install the required dependencies:
```sh
pip install -r requirements.txt
```
3. Run the Streamlit app:
```sh
streamlit run app.py
```
4. Upload a clear image of a tomato leaf
5. Review the AI's diagnosis and recommendations

## Model Information

The app uses a pre-trained Vision Transformer (ViT) model fine-tuned on a dataset of tomato leaf images. The model can identify the following conditions:

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy Plants

## Contributing

Contributions to improve the Tomato Disease Detective are welcome! Please feel free to submit a Pull Request.

## Contact

For any queries or suggestions, please open an issue in this repository.

Happy Gardening! 🌱
