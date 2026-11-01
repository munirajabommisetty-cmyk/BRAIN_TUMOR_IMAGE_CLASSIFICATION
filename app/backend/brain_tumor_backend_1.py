from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import base64

app = FastAPI(
    title="Brain Tumor Classification API",
    description="CNN prediction with Grad-CAM visualization",
    version="1.0.0"
)

# --------------------------------------------------
# 1. LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "brain_tumor_pred.keras"
IMAGE_SIZE = (224, 224)
TARGET_LAYER_NAME = "conv2d_5"

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# --------------------------------------------------
# 2. BUILD GRAD-CAM MODEL
# --------------------------------------------------

def build_grad_model(model, target_layer_name):
    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = inputs
    target_activation = None

    # Rebuild the Sequential model using its existing layers.
    # This avoids relying on model.output in Keras 3.
    for layer in model.layers:
        x = layer(x)

        if layer.name == target_layer_name:
            target_activation = x

    if target_activation is None:
        available_layers = [layer.name for layer in model.layers]

        raise ValueError(
            f"Target layer '{target_layer_name}' not found. "
            f"Available layers: {available_layers}"
        )

    return tf.keras.Model(
        inputs=inputs,
        outputs=[target_activation, x]
    )


grad_model = build_grad_model(model, TARGET_LAYER_NAME)


# --------------------------------------------------
# 3. GENERATE GRAD-CAM
# --------------------------------------------------

def generate_gradcam(image_array):
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            image_array,
            training=False
        )

        predicted_index = tf.argmax(
            predictions[0],
            output_type=tf.int32
        )

        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    # Average gradients spatially to obtain channel weights.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension.
    feature_maps = feature_maps[0]

    # Weight each feature-map channel.
    weighted_feature_maps = (
        feature_maps * pooled_gradients
    )

    # Combine channels into a 2D heatmap.
    heatmap = tf.reduce_sum(
        weighted_feature_maps,
        axis=-1
    )

    # Keep positive contributions.
    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap.numpy()

    # Normalize only if meaningful positive activation exists.
    heatmap_max = float(heatmap.max())

    if heatmap_max > 1e-8:
        heatmap = heatmap / heatmap_max
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap, predictions.numpy()


# --------------------------------------------------
# 4. COLORIZE HEATMAP
# --------------------------------------------------

def colorize_heatmap(heatmap):
    """
    Convert a normalized grayscale heatmap to RGB
    using a JET-like color mapping.
    """

    x = np.clip(heatmap, 0, 1)

    red = np.clip(
        1.5 - np.abs(4 * x - 3),
        0,
        1
    )

    green = np.clip(
        1.5 - np.abs(4 * x - 2),
        0,
        1
    )

    blue = np.clip(
        1.5 - np.abs(4 * x - 1),
        0,
        1
    )

    colored = np.stack(
        [red, green, blue],
        axis=-1
    )

    return (colored * 255).astype(np.uint8)


# --------------------------------------------------
# 5. ENCODE IMAGE AS BASE64
# --------------------------------------------------

def image_to_base64(image_array):
    image = Image.fromarray(image_array)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# --------------------------------------------------
# 6. HOME AND HEALTH ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Brain Tumor Classification API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# 7. PREDICTION + GRAD-CAM ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Decode and standardize the uploaded image.
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        original_image = np.array(image)

        # Add batch dimension.
        image_array = np.expand_dims(
            original_image.astype(np.float32),
            axis=0
        )

        # The model already includes Rescaling(1./255).
        # Do not divide image_array by 255 here.

        # Generate prediction and Grad-CAM.
        heatmap, predictions = generate_gradcam(
            image_array
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index]
        )

        predicted_class = classes[predicted_index]

        # Resize heatmap to match the MRI image.
        heatmap_image = Image.fromarray(
            np.uint8(255 * heatmap)
        ).resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        heatmap_resized = (
            np.array(heatmap_image).astype(np.float32)
            / 255.0
        )

        # Generate colored heatmap.
        heatmap_color = colorize_heatmap(
            heatmap_resized
        )

        # Overlay heatmap on original MRI.
        overlay = np.clip(
            0.6 * original_image.astype(np.float32)
            + 0.4 * heatmap_color.astype(np.float32),
            0,
            255
        ).astype(np.uint8)

        # Encode visualization images for JSON.
        heatmap_base64 = image_to_base64(
            heatmap_color
        )

        overlay_base64 = image_to_base64(
            overlay
        )

        # Keep these response keys synchronized
        # with the Streamlit frontend.
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "gradcam_heatmap": heatmap_base64,
            "gradcam_overlay": overlay_base64
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )