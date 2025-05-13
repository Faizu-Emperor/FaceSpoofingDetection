# 🧠 Face Spoofing Detection using Deep Learning (YOLOv8)

This project implements a **real-time face spoofing detection system** that classifies live video input as either **real** or **fake** using a YOLOv8-based deep learning model.  
The aim is to prevent unauthorized access through spoofing attacks such as photos, videos, or masks.

---

## 📁 About the Dataset

The dataset used includes a mixture of **real** and **fake** face images, collected from various sources and manually labeled.  
Each sample includes:

- `.jpg` image of the face  
- `.txt` annotation file in YOLO format

The dataset was split into:

- **Training**: 70%  
- **Validation**: 20%  
- **Testing**: 10%  

using a custom `splitData.py` script.

---

## ⚙️ Dataset Processing

- **Resizing**: All images resized to `416x416` for YOLOv8 compatibility  
- **Labeling**: YOLO-format bounding box labels for `real` and `fake` classes  
- **Data YAML**: Auto-generated `data.yaml` file for model configuration

---

## 🛠️ Data Collection

A custom script `dataCollection.py` was used to:

- Capture real-time face images via webcam  
- Store and label them as either `real` or `fake`  
- Save annotated image and label pairs for training

---

## 🧠 Model Architecture & Training

We used the **YOLOv8 Medium** model (`yolov8m.pt`) from [Ultralytics](https://github.com/ultralytics/ultralytics) for fast yet accurate face spoof detection.

### 🔧 Training Configuration (`train.py`)

| Parameter       | Value         |
|------------------|---------------|
| Epochs           | 150           |
| Batch Size       | 16            |
| Image Size       | 416x416       |
| Learning Rate    | 0.0005        |
| Patience         | 10 (early stopping) |
| Device           | CUDA (GPU)    |

## 🚀 Real-Time Inference

The `main.py` script runs the YOLOv8 model on a live webcam feed and performs real-time classification:

- 📦 Detects face bounding boxes  
- 🧠 Classifies each face as **Real** or **Fake**  
- 📊 Displays the label and confidence score overlayed on the video

## 📊 Results & Performance

| Metric             | Value (Approx.) |
|--------------------|-----------------|
| **Accuracy**        | ~98%            |
| **Precision (Real)**| ~99.5%          |
| **Precision (Fake)**| ~96–98%         |
| **Recall (Real)**   | ~98–99%         |
| **Recall (Fake)**   | ~95–97%         |
| **FPS (Live)**      | 20–30           |

> ✅ The model performs with **high confidence** in real-time environments, accurately identifying spoofed images while maintaining fast inference speeds.

## 🧪 Testing

The system has been tested on various spoofing inputs, including:

- 🖼️ **Printed photos**
- 📱 **Screen videos**
- 🎭 **Masked faces**

All tests were performed under **varied lighting conditions** and showed **consistent and reliable results**.

## 👨‍💻 Contributors

- **Faiz Ali**
- **Yeshwanth Reddy**
- **Shiva Reddy**

---

## 📌 Future Scope

- 🔁 **Expand Dataset**: Include more spoofing variations (e.g., 3D masks)  
- 📱 **Mobile Deployment**: Convert model using TensorFlow Lite or ONNX  
- 🎥 **Multiple Face Detection**: Extend to detect multiple spoofed faces  
- 🛡️ **Liveness Detection**: Integrate additional temporal/spatial features
