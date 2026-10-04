import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DigitVision AI",
    page_icon="🔢",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    background: #eef6ff;
}

.digit {
    font-size: 70px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")


model = load_model()


# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🔢 DigitVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Handwritten Digit Recognition using Deep Learning</div>',
    unsafe_allow_html=True
)


# ---------------- INFO ----------------
st.markdown("""
<div class="card">
<h3>🧠 About the Project</h3>
<p>
This application uses a Convolutional Neural Network (CNN) trained on
the MNIST dataset to recognize handwritten digits from 0 to 9.
</p>
</div>
""", unsafe_allow_html=True)


# ---------------- UPLOAD ----------------
st.markdown(
    '<div class="card"><h3>📤 Upload Your Digit</h3>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.markdown("### 🖼️ Uploaded Image")

    st.image(
        image,
        width=250
    )

    # Convert image
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    # ---------------- RESULT ----------------
    st.markdown("""
    <div class="result">
    <h2>🎯 Prediction Result</h2>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="digit">{predicted_digit}</div>',
        unsafe_allow_html=True
    )

    st.success(f"Predicted Digit: {predicted_digit}")

    st.write(f"Confidence: **{confidence:.2f}%**")

    st.progress(min(confidence / 100, 1.0))

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- MODEL INFO ----------------
st.markdown("### 📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "CNN")

with col2:
    st.metric("Dataset", "MNIST")

with col3:
    st.metric("Accuracy", "99.04%")


# ---------------- HOW IT WORKS ----------------
with st.expander("🔍 How does this project work?"):

    st.write("""
    1. User uploads a handwritten digit image.
    2. The image is converted into grayscale.
    3. The image is resized to 28 × 28 pixels.
    4. Pixel values are normalized between 0 and 1.
    5. The CNN model processes the image.
    6. The model predicts a digit from 0 to 9.
    7. The application displays the predicted digit and confidence.
    """)


# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">Built with Python • TensorFlow • CNN • Streamlit</div>',
    unsafe_allow_html=True
)