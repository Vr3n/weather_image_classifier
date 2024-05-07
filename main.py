import PIL.ImageOps
import streamlit as st
import PIL
import tensorflow as tf
import io
import numpy as np
import pandas as pd
import warnings
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Image Weather Classifier.",
    page_icon=":cloud:",
    initial_sidebar_state='auto'
)

hide_streamlit_style = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the css

with st.sidebar:
    st.title("Weather Classifier App.")
    st.subheader("Detection of weather from the photos of landscapes.")

    uplaoded_img = st.file_uploader("Upload Your Image", type=[
                    "png", 'jpg', 'jpeg'], key="img_uploader")


model = tf.keras.models.load_model("best_model")


def import_and_predict(image_data, model):
    size = (160, 160)
    img = img_to_array(image_data)
    if img.shape[2] == 1:
        img = np.concatenate((img_to_array(img),) * 3, axis=2)
    img = tf.image.resize(img, size)
    img = img / 255
    img = img[None,...]
    prediction = model.predict(img)
    return prediction

def calculate_prediction(prediction, class_labels):
    max_prob_index = np.argmax(prediction)
    pred_percentage = dict()

    for i, prob in enumerate(prediction.flatten()):
        class_name = class_labels[i]
        percentage = prob * 100
        pred_percentage[class_name] = percentage

    return pred_percentage, max_prob_index



if uplaoded_img is None:
    st.markdown("## 👈 Please upload an image.")
else:
    image = PIL.Image.open(uplaoded_img)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_labels = ["Cloudy", "Rainy", "Sunny"]
    class_percentage, max_prob_index = calculate_prediction(predictions, class_labels)
    st.markdown(f"### The weather is {class_labels[max_prob_index]}.")
    st.text("Percentage of weather detected by the model.")
    st.dataframe(class_percentage, use_container_width=True)
