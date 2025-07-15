import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("mnist_cnn_model.keras")

# Streamlit UI setup
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🔢 Handwritten Digit Recognition")
st.markdown("Draw a digit or upload an image to predict (0-9)")

# Tabs for drawing and uploading
option = st.radio("Choose input method:", ["🖌️ Draw Digit", "📁 Upload Image"])

image = None

if option == "🖌️ Draw Digit":
    st.markdown("### Draw below 👇")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        image = canvas_result.image_data

elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=150)

# Prediction Logic
def preprocess_and_predict(img):
    if img is None:
        return "Please draw or upload a digit first."

    # Convert to grayscale numpy array
    img = np.array(img)

    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert if background is white and digit is black
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize and reshape
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(img)
    return f"🧠 Predicted Digit: **{np.argmax(pred)}**"

# Predict Button
if st.button("🔍 Predict"):
    result = preprocess_and_predict(image)
    st.success(result)
