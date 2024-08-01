import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

st.header('Image Classification Model')

# Load the model
model = load_model('/home/codename/projects/learning/machine_learning/trials/fruit_classification/Image_classify.keras')

# Image categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Image dimensions
img_height = 180
img_width = 180

# File uploader
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image
    image_path = os.path.join("temp", uploaded_file.name)
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))

    # Preprocess the image
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display the image and prediction
    st.image(image_load, width=200)
    st.write('The image is a: ' + data_cat[np.argmax(score)])
    accuracy = np.max(score)*100
    rounded = round(accuracy, 2)
    st.write('With accuracy of ' + str(rounded))
