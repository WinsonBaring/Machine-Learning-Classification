import streamlit as st
from functions import load_images_and_labels 
from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from camera_input_live import camera_input_live
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# Title


st.title('Black and Yellow Classification')

tiger_label = [0, 0, 1]
tiger_images_validation, tiger_labels_validation = load_images_and_labels('data/tiger_validation_resized',tiger_label)


# Load the Keras model
model = load_model('./models/classification_model.h5')
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary cross-entropy loss for multi-label classification
              metrics=['accuracy'],
              )
loss,accuracy = model.evaluate(tiger_images_validation,tiger_labels_validation)

def process_image(image_array):
    # Convert the image buffer to a numpy array
    nparr = np.frombuffer(image_array.getvalue(), np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_image = cv2.resize(cv2_img, (100, 100))
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Main Streamlit app
def main():
    st.title("Live Camera Classification")

    # Display the camera input
    img_file_buffer = camera_input_live()
    
    # img_file_buffer = st.camera_input("Take a picture")

    # Check if an image was captured
    if img_file_buffer is not None:
        st.image(img_file_buffer)

        # Preprocess the image
        processed_image = process_image(img_file_buffer)

        # Make predictions using the model
        predicted = model.predict(processed_image)

        # Display the predicted classification
        st.write("Predicted Classification:", predicted)
        

# Run the Streamlit app
if __name__ == "__main__":
    main()




