import streamlit as st
from functions import load_images_and_labels 
from keras.models import load_model
from tensorflow.keras.models import load_model,Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from camera_input_live import camera_input_live
import matplotlib.pyplot as plt
import numpy as np
# import cv2

import streamlit as st
from PIL import Image

import os
# Title


st.markdown("<h1 style='text-align: center;'>The Identity Your Animal</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # Add space below the title

# st.image('./data/bg.png', use_column_width=True)
# tiger_label = [0, 0, 1]
# # tiger_images_validation, tiger_labels_validation = load_images_and_labels('data/tiger_validation_resized',tiger_label)



# # loss,accuracy = model.evaluate(tiger_images_validation,tiger_labels_validation)

# # Title for the app
# st.markdown("<h1 style='text-align: center;'>Instruction</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Upload an image of either Lion, Tiger, Hyeena, Cheetah and the app will identify the result</p>", unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)  # Add space below the title
# st.markdown("<br>", unsafe_allow_html=True)  # Add space below the title

# # Image uploader
# uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# # Load the Keras model
# model = load_model('classification_model.h5')
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',  # Binary cross-entropy loss for multi-label classification
#               metrics=['accuracy'],
#               )
# # Function to make prediction
# def predict(image):
#     # Resize image to match model input shape
#     image = image.resize((100, 100))
#     # Convert image to numpy array
#     image_array = np.array(image)
#     # Normalize pixel values
#     image_array = image_array / 255.0
#     # Add batch dimension
#     image_array = np.expand_dims(image_array, axis=0)
#     # Perform prediction using the model
#     prediction = model.predict(image_array)
#     accuracy = .7
#     st.write('Prediction1:',prediction)
#     st.write('Prediction2:',prediction[[[0]]])
#     st.write('Prediction2:',prediction[[[[0]]]])
#     st.write('Prediction3:',prediction[[[[[0]]]]] )
    



# # Display the uploaded image
# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Button to trigger prediction
#     if st.button('Predict'):
#         predict(image)
# else:
#     st.write('No image uploaded yet.')

# def process_image(image_array):
#     # Convert the image buffer to a numpy array
#     nparr = np.frombuffer(image_array.getvalue(), np.uint8)
#     cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     processed_image = cv2.resize(cv2_img, (100, 100))
#     processed_image = processed_image / 255.0
#     processed_image = np.expand_dims(processed_image, axis=0)
#     return processed_image

# # Main Streamlit app
# def main():
#     st.title("Live Camera Classification")

#     # Display the camera input
#     img_file_buffer = camera_input_live(debounce=500)
    
#     # img_file_buffer = st.camera_input("Take a picture")

#     # Check if an image was captured
#     if img_file_buffer is not None:
#         st.image(img_file_buffer)

#         # Preprocess the image
#         processed_image = process_image(img_file_buffer)

#         # Make predictions using the model
#         predicted = model.predict(processed_image)

#         # Display the predicted classification
#         st.write("Predicted Classification:", predicted)
        

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()




