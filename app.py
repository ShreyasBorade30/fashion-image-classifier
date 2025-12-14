import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("fashion_mnist_model.h5")

# Class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

st.title("Fashion MNIST Image Classifier 👕👗👟")

st.write("Upload a grayscale image (28×28 or any size — I will preprocess it).")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess
    img = image.resize((28, 28))   # resize to match Fashion MNIST
    img_array = np.array(img)

    img_array = img_array / 255.0  # normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # model expects (batch, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction[0])
    confidence = prediction[0][class_id] * 100

    st.subheader(f"Prediction: **{class_names[class_id]}**")
    st.write(f"Confidence: {confidence:.2f}%")
