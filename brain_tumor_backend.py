from fastapi import FastAPI, File, UploadFile
#fastapi  is used to create the api application.
from PIL import Image#pillow is used to work with images.
import tensorflow as tf#for cnn.
import numpy as np

app = FastAPI()# api creation

# Load trained CNN
model = tf.keras.models.load_model("brain_tumor_pred.keras")


classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


@app.get("/")#retrive the data.
def home():
    return {"message": "Brain Tumor Classification API"}


@app.post("/predict")#predict api endpoint.
async def predict(file: UploadFile = File(...)):#asynchorouns function to do parallel workflow.

    # 1. Read uploaded image
    image = Image.open(file.file).convert("RGB")

    # 2. Resize to the size used during training
    image = image.resize((224, 224))

    # 3. Convert image to NumPy array
    image_array = np.array(image)

    # 4. Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # 5. Prediction
    prediction = model.predict(image_array)

    # 6. Get predicted class
    predicted_index = np.argmax(prediction[0])

    # 7. Get confidence
    confidence = float(prediction[0][predicted_index])

    return {
        "predicted_class": classes[predicted_index],
        "confidence": confidence
    }