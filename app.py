import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()
st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below and click Predict:")

canvas_result = st_canvas(
    fill_color="#FFFFFF",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict") and canvas_result.image_data is not None:
    img = canvas_result.image_data
    im = Image.fromarray((img[:, :, :3]).astype('uint8'), 'RGB')
    im = ImageOps.grayscale(im).resize((28, 28))
    im = ImageOps.invert(im)
    x = np.array(im).astype("float32") / 255.0
    x = x.reshape(1, 28, 28)
    preds = model.predict(x)
    digit = np.argmax(preds, axis=1)[0]
    st.write(f"**I think you drew:** {digit}")
