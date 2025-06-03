# Fall Detection System using CNN and YOLOv8   

## 📌 Overview  
A hybrid fall detection system combining **Convolutional Neural Networks (CNN)** for posture classification and **YOLOv8** for real-time object detection. Designed to identify critical fall incidents and trigger immediate alerts.  

---

## ✨ Features  
- **Multi-class CNN Classifier**: Detects `sitting`, `fall`, `not_fall`, and `bend` postures.  
- **Real-time Alerts**: Automatic fall detection with emergency notifications.  
- **Data Augmentation**: Enhanced training with rotations, flips, and brightness adjustments.  
- **Comprehensive Metrics**: Accuracy/loss curves, confusion matrix, and classification reports.  
- **YOLOv8 Integration**: (Referenced) Optional object detection pipeline.  

---
⚙️ Training Configuration
Parameter	Value
Optimizer	Adam (lr=0.0001)
Loss Function	Categorical Crossentropy
Epochs	50
Batch Size	32
Image Size	224x224

## Requirements
Python >= 3.6
tensorflow >= 2.0
opencv-python
numpy
matplotlib
seaborn
scikit-learn
ultralytics (for YOLOv8)

## 🌟 Future Work
Integrate YOLOv8 for real-time person detection.
Deploy as a Flask/Django web service.
Optimize model for edge devices (TensorFlow Lite).

## 🗂 Dataset Structure  
```plaintext
dataset/  
├── train/  
│   ├── sitting/  
│   ├── fall/  
│   ├── not_fall/  
│   └── bend/  
└── test/  
    ├── sitting/  
    ├── fall/  
    ├── not_fall/  
    └── bend/

🧠 Model Architecture
CNN Classifier:
Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(512, (3,3), activation='relu'),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])
