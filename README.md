# Brain_MRI-segmentation

This project implements a deep learning model to perform **brain tumor segmentation** on MRI images using a custom **U-Net architecture**. It includes a clean, interactive **Streamlit web app** for easy image upload and live mask prediction.

---


## 📌 Features

- ✅ Brain MRI segmentation using a custom U-Net model.
- ✅ Streamlit UI to upload images and visualize outputs.
- ✅ Custom loss and metrics: Dice Coefficient, IoU.
- ✅ Live prediction and display of segmentation masks.
- ✅ Data augmentation to improve generalization.

---

## 🧠 Model Architecture

- **Base Model**: U-Net (custom implementation)
- **Input Shape**: `(256, 256, 3)`
- **Loss Function**: Dice Coefficient Loss
- **Optimizer**: Adam (`learning_rate=1e-4`)
- **Metrics**: Binary Accuracy, Dice Coefficient, IoU

---

## 📈 Training Details

- **Epochs**: 100
- **Batch Size**: 32
- **Data Augmentation**:
  - Rotation (±20%)
  - Width & height shift (±5%)
  - Shear and zoom
  - Horizontal flip

- **Model Checkpoint**: Saves best model as `unet_brain_mri_seg.h5`

---

## 🧪 Evaluation Metrics

| Metric              | Description                                                      |
|---------------------|------------------------------------------------------------------|
| **Dice Coefficient** | Measures overlap between predicted and ground truth masks        |
| **IoU (Jaccard)**    | Ratio of intersection over union between prediction and truth    |
| **Binary Accuracy**  | Pixel-wise accuracy of the segmentation mask                     |

![Screenshot 2025-06-10 185623](https://github.com/user-attachments/assets/c18d3fe4-22a8-47da-8542-47935d4a39ad)

---

## 🛠 Tech Stack

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit 🚀


