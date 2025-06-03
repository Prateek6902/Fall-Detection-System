
#import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Define paths to train and test datasets
train_dir = 'C:/Users/ASUS/Desktop/YoloV8/dataset'
test_dir = 'C:/Users/ASUS/Desktop/YoloV8/test_data'

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Using larger image size for better feature extraction
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to keep shuffle=False for correct mapping of predictions
)

# Updated CNN model with deeper layers
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 classes: sitting, fall, not_fall, bend
])

# Compile the model with a tuned learning rate and optimizer
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # Adam optimizer with a low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the CNN model with more epochs for better learning
history = model.fit(
    train_generator,
    epochs=50,  # More epochs for fine-tuning
    validation_data=test_generator
)

# Plot the train-test curve for accuracy and loss
def plot_train_test_curve(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Call the function to plot the curves
plot_train_test_curve(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Predict the test set labels
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test generator
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Check for fall incidents and print an alert message if detected
fall_class_index = class_names.index('fall')  # Assuming 'fall' is the label for fall incidents

# Loop through the predictions to check for fall incidents
for i, pred in enumerate(y_pred):
    if pred == fall_class_index:
        print(f"Alert: Fall incident detected for image {i + 1}. Please provide assistance immediately!")

# Print the confusion matrix
conf_matrix = confusion_matrix(true_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print("Classification Report:\n", classification_report(true_labels, y_pred, target_names=class_names))

# Scatter Plot for Predicted vs. True Labels
plt.figure(figsize=(10, 8))
scatter_colors = ['red', 'blue', 'green', 'orange']

for i, class_name in enumerate(class_names):
    indices = np.where(true_labels == i)
    plt.scatter(true_labels[indices], y_pred[indices], color=scatter_colors[i], label=class_name, alpha=0.6)

plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Scatter Plot of True vs Predicted Labels')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names, rotation=45)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Box Plot of Prediction Probabilities for Each Class
plt.figure(figsize=(12, 8))
prediction_probabilities = [Y_pred[:, i] for i in range(len(class_names))]
sns.boxplot(data=prediction_probabilities)
plt.xlabel('Classes')
plt.ylabel('Prediction Probabilities')
plt.title('Box Plot of Prediction Probabilities for Each Class')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.grid(True)
plt.show()

# Display a few test images with their predicted and true labels
def display_predictions(test_generator, predictions, true_labels, class_names, num_images=15):
    plt.figure(figsize=(15, 10))
    count = 0

    for i in range(len(test_generator)):
        img, label = test_generator[i]
        batch_size = img.shape[0]

        for j in range(batch_size):
            if count >= num_images:
                break
            plt.subplot(3, 5, count + 1)
            plt.imshow(img[j])
            true_label = class_names[true_labels[count]]
            predicted_label = class_names[predictions[count]]
            plt.title(f'True: {true_label}\nPred: {predicted_label}')
            plt.axis('off')
            count += 1

        if count >= num_images:
            break

    plt.tight_layout()
    plt.show()

# Display some test images with their true and predicted labels
display_predictions(test_generator, y_pred, true_labels, class_names, num_images=15)
