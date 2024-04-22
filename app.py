import streamlit as st


# Title
st.title('My First Streamlit App')

# Header
st.header('Welcome to Streamlit!')

# Text
st.write('This is a simple example of a Streamlit app.')

# Add a slider
age = st.slider('Select your age', 0, 100, 25)

# Show the selected age
st.write('Your selected age:', age)

image = st.camera()

# Display captured image
if image is not None:
    st.image(image, caption='Captured Image')
else:
    st.write('No image captured')


