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


## 🗂 Dataset Structure  
```plaintext
dataset/  
├── train/  
│   ├── sitting/  
│   ├── fall/  
│   ├── not_fall/  
│   └── bend/  
└── test/  
    ├── sitting/![fall-10-cam0-rgb-026](https://github.com/user-attachments/assets/de83b864-49f6-4d49-bcbd-42b8796e5795)
    ├── fall/![fall-03-cam0-rgb-206](https://github.com/user-attachments/assets/0cbd5616-d132-4ef7-a45b-fcf60b2c5609)
    ├── not_fall/![fall-09-cam0-rgb-011](https://github.com/user-attachments/assets/ea332f3e-e445-4413-940c-5b921c2696e0)
    └── bend/![fall-01-cam0-rgb-097](https://github.com/user-attachments/assets/c0b444fe-a14e-4ad7-a1d0-377e80493219)
```    
##🧠 Model Architecture
CNN Classifier:
```Sequential([
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
```
📊 Evaluation Metrics
1. Accuracy/Loss Curves
![test_train_curve](https://github.com/user-attachments/assets/ed349c08-cd1d-45b8-af51-aa90504ce66f)
![Figure 2024-10-29 161222](https://github.com/user-attachments/assets/c630b506-543d-4dad-a8e3-851965520877)

2. Confusion Matrix
```
confusion_matrix(true_labels, predictions)
![Figure 2024-10-29 161215](https://github.com/user-attachments/assets/ad4ccde4-b3f2-4879-a1a9-fe356603109e)
![Figure 2024-10-09 160122](https://github.com/user-attachments/assets/602bda68-783e-42eb-b705-a2d2ce9305d8)

```
3. Sample Alerts
```
Alert: Fall incident detected for image 42. Please provide assistance immediately!
```
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
