import streamlit as st
import requests#it is used to handle the post requests.
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI image for classification.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file :

    # Display uploaded image using the pillow.
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            # Send image to FastAPI
            files = {
                "file": ( uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            response = requests.post("https://brain-tumor-image-classification-1.onrender.com/predict",files=files)

        # Check response
        if response.status_code == 200:

            result = response.json()

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        else:

            st.error(
                f"Prediction failed: {response.text}"
            )
