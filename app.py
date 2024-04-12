import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mnist_cnn.h5")

st.title("✍️ Draw a Digit")
st.markdown("Draw a digit (0–9) below and let the model guess it!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # black ink
    stroke_width=10,
    stroke_color="#000000",  # black stroke
    background_color="#FFFFFF",  # white background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


if canvas_result.image_data is not None:
    # Convert image to grayscale PIL image
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"), mode="L")
    img = img.resize((28, 28)).convert("L")  # resize to 28x28 and ensure grayscale

    # Normalize and reshape
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"🧠 Model Prediction: **{predicted_digit}**")
