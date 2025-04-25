# Satellite Image Classification - Cloudy vs Water

This project is a deep learning-based image classification task that distinguishes between two types of satellite images: **cloudy** and **water**. The model is built using TensorFlow and trained on a custom dataset.

## 📁 Dataset

The dataset contains images categorized into two classes:

- `cloudy/`: Satellite images covered with clouds.
- `water/`: Satellite images showing clear water surfaces.

The dataset is divided into:
- Training set
- Validation set
- Test set

## 🧠 Model Architecture

The model uses **Transfer Learning** with a pre-trained base (e.g., `MobileNetV2`), followed by custom layers:

```python
model_1 = Sequential()
 
model_1.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150,150,1)))
model_1.add(BatchNormalization())
model_1.add(MaxPool2D((2, 2)))
 
model_1.add(Conv2D(32, (4, 4),padding='same', activation='relu'))
model_1.add(BatchNormalization())
model_1.add(MaxPool2D((2, 2)))
 
model_1.add(Conv2D(32, (7, 7), padding='same', activation='relu'))
model_1.add(BatchNormalization())
model_1.add(MaxPool2D((2, 2)))
 
model_1.add(Flatten())
model_1.add(Dense(128, activation = 'relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(64, activation = 'relu'))
model_1.add(Dropout(0.3))
 
model_1.add(Dense(1, activation='sigmoid'))
```

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The target accuracy for both training and testing is **≥ 85%**.

## 🧪 Tools & Libraries

- Python
- TensorFlow / Keras
- Matplotlib / Seaborn
- Scikit-learn

## 💾 Saved Model Formats

The trained model is exported in multiple formats:

```
klasifikasi gambat/
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
```

## 🚀 How to Use

You can run the model for prediction using any of the exported formats:
- In the browser using TensorFlow.js
- On edge devices using TensorFlow Lite
- As a standard `.h5` or `SavedModel` in Python
