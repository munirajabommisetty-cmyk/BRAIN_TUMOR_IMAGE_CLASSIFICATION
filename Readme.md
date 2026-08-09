# 🧠 Brain Tumor Classification using CNN

A deep learning-based image classification system that classifies brain MRI images into four categories using a Convolutional Neural Network (CNN).

The project includes:

* 🧠 CNN-based brain MRI classification
* ⚡ FastAPI backend for model inference
* 🎨 Streamlit frontend for image upload and prediction
* 📊 Model evaluation using accuracy, loss, precision, recall, F1-score, and classification report
* 🖼️ Support for JPEG, JPG, and PNG image uploads
* 🔌 REST API for prediction

---

## 📌 Project Overview

Brain tumor classification from MRI images is an important computer vision application where deep learning can be used to identify different tumor categories.

In this project, a CNN is trained to classify brain MRI images into four classes:

| Class        | Description             |
| ------------ | ----------------------- |
| `glioma`     | Glioma tumor            |
| `meningioma` | Meningioma tumor        |
| `no_tumor`   | Brain MRI without tumor |
| `pituitary`  | Pituitary tumor         |

The trained model is saved as a Keras model:

```text
brain_tumor_pred.keras
```

The model is then integrated with FastAPI to provide predictions through a REST API, while Streamlit provides a simple user interface for uploading MRI images.

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               │ Upload MRI
                               ▼
                    ┌─────────────────────┐
                    │     Streamlit       │
                    │     Frontend        │
                    └──────────┬──────────┘
                               │
                               │ HTTP POST
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    │                     │
                    │ RGB conversion      │
                    │ Resize → 224×224    │
                    │ NumPy conversion    │
                    │ Batch dimension     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     CNN Model       │
                    │ brain_tumor_pred    │
                    │      .keras         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Prediction       │
                    │                     │
                    │ Class + Confidence  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Streamlit       │
                    │ Display Result      │
                    └─────────────────────┘
```

---

# 📂 Project Structure
```text
brain-tumor-classification/
│
├── dataset/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
|
|── brain_tumor_classification.ipynb
│
├── brain_tumor_pred.keras
│
├── brain_tumor_backend.py
│
├── brain_tumor_fronetend.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```
---

# 🔬 Deep Learning Workflow

The project follows the standard deep learning image classification workflow:

```text
Dataset
   ↓
Image Loading
   ↓
Image Preprocessing
   ↓
Train / Validation / Test
   ↓
CNN Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
FastAPI Deployment
   ↓
Streamlit Frontend
```

---

# 🖼️ Image Preprocessing

The CNN expects images with a spatial resolution of:

```text
224 × 224
```

and three color channels:

```text
RGB
```

Therefore, uploaded images are processed before inference.

### 1. Convert image to RGB

```python
image = Image.open().convert("RGB")
```

This ensures that images with different modes such as grayscale or RGBA are converted into three-channel RGB images.

### 2. Resize image

```python
image = image.resize(( , ))
```

The image is resized to match the input dimensions expected by the CNN.

### 3. Convert to NumPy

```python
image_array = np.array(image)
```

The PIL image is converted into a numerical array.

### 4. Add batch dimension

```python
image_array = np.expand_dims(image_array, axis=0)
```

The resulting shape becomes:

```text
(1, 224, 224, 3)
```

where:

```text
1    → number of images
224  → height
224  → width
3    → RGB channels
```

---

# 🧠 CNN Classification

The trained CNN produces probabilities for the four classes.

For example:

```text
glioma       → 0.10
meningioma   → 0.75
no_tumor     → 0.05
pituitary    → 0.10
```

The class with the highest predicted probability is selected using:

```python
predicted_index = np.argmax(prediction[0])
```

The class index is then mapped to its corresponding class name.

```python
classes = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]
```

---

# 📊 Model Performance

The model achieved approximately:

```text
Accuracy: 0.89
```
on the evaluation set containing 1,000 images.

### Training Performance

| Epoch | Accuracy |   Loss | Validation Loss | Validation Accuracy |
| ----: | -------: | -----: | --------------: | ------------------: |
|     0 |   0.6950 | 0.7750 |          0.5337 |               0.782 |
|     1 |   0.8708 | 0.3617 |          0.3107 |               0.889 |
|     2 |   0.9258 | 0.2049 |          0.3353 |               0.896 |
|     3 |   0.9523 | 0.1302 |          0.2495 |               0.925 |
|     4 |   0.9798 | 0.0673 |          0.2694 |               0.919 |

The model reached approximately **97.98% training accuracy** while validation accuracy reached approximately **89.8%** at its best point in the reported epochs.

This indicates that the model learned the training data effectively, although the difference between training and validation performance should be monitored for potential overfitting.

---

# 📈 Classification Report

The model was evaluated on 1,000 test/evaluation samples.

| Class                | Precision |   Recall | F1-Score |  Support |
| -------------------- | --------: | -------: | -------: | -------: |
| Glioma               |      0.92 |     0.81 |     0.86 |      254 |
| Meningioma           |      0.91 |     0.80 |     0.85 |      306 |
| No Tumor             |      0.87 |     0.99 |     0.93 |      140 |
| Pituitary            |      0.86 |     1.00 |     0.93 |      300 |
| **Accuracy**         |           |          | **0.89** | **1000** |
| **Macro Average**    |  **0.89** | **0.90** | **0.89** | **1000** |
| **Weighted Average** |  **0.89** | **0.89** | **0.89** | **1000** |

---

# 🔎 Performance Analysis

### Glioma

```text
Precision: 0.92
Recall:    0.81
F1-score:  0.86
```

The model has high precision for glioma, meaning that when it predicts glioma, the prediction is generally reliable. However, the recall of 0.81 indicates that some actual glioma images are classified as other classes.

### Meningioma

```text
Precision: 0.91
Recall:    0.80
F1-score:  0.85
```

The model performs reasonably well but misses some meningioma cases.

### No Tumor

```text
Precision: 0.87
Recall:    0.99
F1-score:  0.93
```

The very high recall indicates that the model identifies most no-tumor samples correctly.

### Pituitary

```text
Precision: 0.86
Recall:    1.00
F1-score:  0.93
```

The model correctly identifies all pituitary samples in the evaluation set according to the reported recall.

---

# ⚠️ Important: Confidence vs Accuracy

The prediction confidence returned by the CNN should **not automatically be interpreted as medical certainty**.

For example:

```text
Prediction: Glioma
Confidence: 94%
```

does not mean:

```text
94% probability that the patient has glioma
```

It represents the model's confidence/probability score for the predicted class under the model's learned classification setup.

This project is intended for **educational and research purposes** and should not be used as a medical diagnosis system.

---

# 🚀 FastAPI Backend

FastAPI is used to expose the trained CNN as a REST API.

The main prediction endpoint is:

```text
POST /predict
```

The API accepts an uploaded image and returns the predicted class and confidence.

Example response:

```json
{
    "predicted_class": "meningioma",
    "confidence": 0.91
}
```

---

# 🔌 FastAPI Prediction Pipeline

The backend performs:

```text
Uploaded Image
      ↓
PIL Image
      ↓
RGB Conversion
      ↓
Resize to 224×224
      ↓
NumPy Array
      ↓
Add Batch Dimension
      ↓
CNN Prediction
      ↓
Argmax
      ↓
Class Mapping
      ↓
JSON Response
---

# 🎨 Streamlit Frontend

Streamlit provides a simple interface for users to upload an image.

The frontend:

1. Allows image upload
2. Displays the uploaded image
3. Sends the image to FastAPI
4. Receives the prediction
5. Displays the predicted class
6. Displays confidence

The architecture is:

```text
Streamlit
    │
    │ requests.post()
    ▼
FastAPI
    │
    ▼
CNN
    │
    ▼
JSON
    │
    ▼
Streamlit
```

---

# 🖼️ Supported Image Formats

The Streamlit interface allows:

```text
.jpg
.jpeg
.png
```

The backend converts uploaded images to RGB:

```python
Image.open(file.file).convert("RGB")
```

Therefore, the CNN receives standardized three-channel image data.

The important requirement is not the original file extension but that the final preprocessing matches the preprocessing used during model training.

---

# 🛠️ Technologies Used

| Technology         | Purpose                       |
| ------------------ | ----------------------------- |
| Python             | Programming language          |
| TensorFlow / Keras | CNN development and inference |
| NumPy              | Numerical operations          |
| Pillow             | Image processing              |
| FastAPI            | REST API backend              |
| Uvicorn            | ASGI server                   |
| Streamlit          | Web frontend                  |
| Scikit-learn       | Model evaluation              |
| Matplotlib         | Visualization                 |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/munirajabommisetty-cmyk/Brain_Tumor_Classification.git
```

Move into the project:

```bash
cd brain-tumor-classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the FastAPI Backend

Start the FastAPI server:

```bash
uvicorn brain_tumor_backend:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation can be accessed at:

```text
http://127.0.0.1:8000/docs
```

You can use `/docs` to upload an image and test the prediction endpoint directly.

---

# ▶️ Running the Streamlit Frontend

Open another terminal and run:

```bash
streamlit run brain_tumor_frontend.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🔄 Complete Prediction Example

Suppose the user uploads:

```text
brain_mri.png
```

The process is:

```text
brain_mri.png
       ↓
Streamlit
       ↓
HTTP POST /predict
       ↓
FastAPI
       ↓
PIL Image
       ↓
RGB
       ↓
224 × 224
       ↓
NumPy
       ↓
(1, 224, 224, 3)
       ↓
CNN
       ↓
[0.04, 0.91, 0.02, 0.03]
       ↓
argmax = 1
       ↓
meningioma
       ↓
confidence = 0.91
```

The user receives:

```text
Prediction: meningioma
Confidence: 91.00%
```

---

# ⚠️ Handling Unsupported Images

A standard multiclass CNN is a **closed-set classifier**.

The model was trained on:

```text
glioma
meningioma
no_tumor
pituitary
```

It does not automatically understand that an unrelated image, such as a cat, is outside those classes.

Therefore, an unrelated image may still be assigned to one of the four classes.

For example:

```text
Cat image
     ↓
CNN
     ↓
no_tumor: 0.65
     ↓
no_tumor
```

This does not mean the image is actually a brain MRI without a tumor.

Possible future improvements include:

* Confidence thresholding
* Adding an `other`/`unknown` class
* Training an MRI vs non-MRI classifier
* Using an out-of-distribution detection method

A robust production system should perform an input-validity check before applying the tumor classifier.

---

# 🧪 Example API Request

The API expects a multipart file upload.

Example using Python:

```python
import requests

files = {
    "file": (
        "brain.jpg",
        open("brain.jpg", "rb"),
        "image/jpeg"
    )
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    files=files
)

print(response.json())
```

Example response:

```json
{
    "predicted_class": "glioma",
    "confidence": 0.88
}
```

---

# 🧩 Future Improvements

Possible improvements for future versions include:

### Model improvements

* Transfer learning using architectures such as EfficientNet, ResNet, or MobileNet
* Data augmentation
* Hyperparameter tuning
* Learning-rate scheduling
* Early stopping
* Class-weight optimization
* Cross-validation
* Better regularization

### Evaluation improvements

* Confusion matrix
* ROC-AUC
* Precision-recall curves
* Per-class error analysis
* Calibration of confidence scores
* Testing on an independent external dataset

### Deployment improvements

* Docker containerization
* Cloud deployment
* Production ASGI configuration
* API authentication
* Request validation
* Logging
* Health-check endpoint

### Safety improvements

* MRI/non-MRI input validation
* Out-of-distribution detection
* Unknown-image rejection
* Confidence calibration
* External validation before real-world use

---

# 📌 Limitations

This project has several limitations.

1. The CNN is trained only on the classes represented in the training dataset.
2. A standard softmax classifier will generally assign an input to one of its known classes even when the input is unrelated.
3. Model performance depends heavily on the quality and distribution of the training dataset.
4. Test-set performance may not represent performance on MRI scans from different hospitals, scanners, populations, or acquisition protocols.
5. Model confidence should not be interpreted as medical certainty.
6. This project is not a substitute for professional medical diagnosis.

---

# 🎯 Learning Outcomes

Through this project, the following concepts were implemented:

* Image classification using CNNs
* Image preprocessing
* RGB image handling
* Image resizing
* NumPy tensor preparation
* Batch dimensions
* CNN inference
* Multiclass classification
* Softmax prediction
* `argmax` class selection
* Model evaluation
* Precision, recall and F1-score
* FastAPI REST APIs
* File uploads using `UploadFile`
* Streamlit frontend development
* Frontend-backend communication
* REST API based model serving
* Basic ML deployment architecture

---

# 👨‍💻 Author

**Muniraja BOMMISETTY**

CSE(AIML) Student

This project was developed as a deep learning and machine learning deployment project to understand the complete workflow from CNN model development to API-based inference and frontend integration.

---

# 📜 Disclaimer

This project is intended for **educational and research purposes only**.

The predictions generated by this model should not be considered medical advice, diagnosis, or treatment recommendations. Any real medical decision must be made by qualified healthcare professionals using appropriate clinical evaluation and diagnostic procedures.
