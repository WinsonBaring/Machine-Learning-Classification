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
label = 'winsson'
st.camera_input(label, key=None, help=None, on_change=None, args=None, disabled=False, label_visibility="visible")
st.button("Testing", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)