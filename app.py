import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import cv2

plt.style.use("ggplot")

# Metrics
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    total = K.sum(y_true + y_pred)
    return (intersection + smooth) / (total - intersection + smooth)

# Streamlit UI
st.set_page_config(page_title="Brain MRI Segmentation", layout="wide")
st.title("🧠 Brain MRI Segmentation App")
st.markdown("Upload a brain MRI image to generate the **segmentation mask** using a pre-trained U-Net model.")

# Load Model
model = load_model("unet_brain_mri_seg.h5", compile=False)
model.compile(
    optimizer='adam',
    loss=dice_coefficients_loss,
    metrics=['binary_accuracy', iou, dice_coefficients]
)

# Constants
im_height, im_width = 256, 256

# File Upload (single image only)
file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if file:
    content = file.getvalue()
    image = np.asarray(bytearray(content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (im_width, im_height))
    image_normalized = image_resized / 255.0
    input_image = image_normalized[np.newaxis, :, :, :]

    # Predict when button clicked
    if st.button("🧪 Predict Mask"):
        pred_mask = model.predict(input_image)[0]  # shape: (256, 256, 1)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_mask_3ch = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        # Layout: side-by-side display
        st.subheader("Results")
        col1,col2,col3,col4 = st.columns(4)

        with col2:
            st.markdown("**Original Image**")
            st.image(image_resized, width=300, caption="Input MRI")

        with col3:
            st.markdown("**Predicted Mask**")
            st.image(pred_mask_3ch, width=300, caption="Segmentation Output")
