import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def predict_image(img):
    # Preprocess the image
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict class probabilities
    preds = model.predict(img_array)
    
    # Decode predictions
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    return decoded_preds

# Streamlit UI
st.title("ResNet50 Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Perform prediction when button is clicked
    if st.button('Classify Image'):
        # Perform prediction
        predictions = predict_image(img)
        
        # Display predictions
        st.subheader("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i + 1}: {label} ({score:.2f})")
