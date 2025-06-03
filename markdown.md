# Fall Detection System using CNN and YOLOv8  

![System Overview](https://via.placeholder.com/800x400?text=Fall+Detection+System)  

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