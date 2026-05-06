# Helmet Detection System using YOLOv8

## Overview

This project implements a Helmet Detection System using YOLOv8 object detection. The system detects riders wearing helmets and riders without helmets from traffic images and videos.

A Streamlit web application is developed for easy local deployment and visualization.

---

# Features

- Helmet detection
- Non-helmet rider detection
- Image upload support
- Video upload support
- Bounding box visualization
- Detection summary
- Streamlit-based UI

---

# Technologies Used

- Python
- YOLOv8
- OpenCV
- Streamlit
- PyTorch

---

# Project Structure

```text
HELMUT_DETECTION/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── yolov8n.pt
│
├── dataset/
│   ├── data.yaml
│   └── train/
│       ├── images/
│       └── labels/
│
└── runs/
```

---

# Installation

Clone repository:

```bash
git clone <repository_link>
cd HELMUT_DETECTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

The dataset is obtained from Roboflow Universe in YOLOv8 format.

Classes used:
- helmet
- no_helmet

---

# Training the Model

Run the training script:

```bash
python train.py
```

Trained weights will be saved in:

```text
runs/detect/train/weights/
```

---

# Running the Application

Run Streamlit application:

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

# Sample Workflow

1. Train YOLOv8 model
2. Launch Streamlit app
3. Upload image/video
4. View detections

---

# Model Details

- Model: YOLOv8n
- Epochs: 20
- Batch Size: 4
- Image Size: 640

---

# Future Improvements

- Real-time CCTV monitoring
- License plate recognition
- Traffic analytics dashboard
- GPU acceleration
- Improved datasets

---

# Limitations

- Reduced accuracy in low-light conditions
- Small object detection challenges
- CPU inference is slower

---

# Author

Manne Bhargav Sai

SMAI Assignment 3

---

# Acknowledgement

This project used AI-assisted tools such as ChatGPT for debugging assistance, report drafting support, and implementation guidance.