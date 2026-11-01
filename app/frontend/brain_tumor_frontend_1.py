import streamlit as st
import requests
import base64
import os

from PIL import Image
from io import BytesIO


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain MRI Analysis",
    page_icon="🧠",
    layout="wide"
)

API_URL = os.getenv(
    "BRAIN_TUMOR_API_URL",
    "http://127.0.0.1:8000/predict"
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .result-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }

    .result-label {
        color: #cbd5e1;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .result-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
    }

    .section-description {
        color: #94a3b8;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🧠 MRI Analysis")
    st.caption("Brain Tumor Classification")

    st.divider()

    st.subheader("How to use")

    st.markdown(
        """
        1. Upload a brain MRI image.
        2. Click **Analyze MRI**.
        3. Review the predicted class and confidence.
        4. Compare the original image with the Grad-CAM results.
        """
    )

    st.divider()

    st.subheader("About Grad-CAM")

    st.caption(
        "Grad-CAM highlights image regions that influenced "
        "the model's prediction. It is an explanatory "
        "visualization, not a tumor segmentation."
    )


# --------------------------------------------------
# 4. MAIN PAGE
# --------------------------------------------------

st.title("Brain MRI Classification")

st.markdown(
    """
    Upload a brain MRI image to obtain a model prediction
    and visualize the image regions that influenced it.
    """
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 5. IMAGE PREVIEW AND ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded MRI")

    st.image(
        original_image,
        caption=uploaded_file.name,
        width=450
    )

    analyze_button = st.button(
        "Analyze MRI",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        with st.spinner(
            "Analyzing the MRI image..."
        ):

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                # Validate required backend response fields.
                required_keys = [
                    "predicted_class",
                    "confidence",
                    "gradcam_heatmap",
                    "gradcam_overlay"
                ]

                missing_keys = [
                    key for key in required_keys
                    if key not in result
                ]

                if missing_keys:
                    st.error(
                        "The backend response is incomplete. "
                        f"Missing fields: {missing_keys}"
                    )

                    st.json(result)
                    st.stop()

                predicted_class = result[
                    "predicted_class"
                ].replace("_", " ").title()

                confidence = float(
                    result["confidence"]
                )

                # Decode the returned Grad-CAM images.
                heatmap_bytes = base64.b64decode(
                    result["gradcam_heatmap"]
                )

                overlay_bytes = base64.b64decode(
                    result["gradcam_overlay"]
                )

                heatmap_image = Image.open(
                    BytesIO(heatmap_bytes)
                ).convert("RGB")

                overlay_image = Image.open(
                    BytesIO(overlay_bytes)
                ).convert("RGB")


                # ----------------------------------
                # 6. PREDICTION SUMMARY
                # ----------------------------------

                st.divider()
                st.header("Prediction Results")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">
                                Predicted Class
                            </div>
                            <div class="result-value">
                                {predicted_class}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col2:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">
                                Model Confidence
                            </div>
                            <div class="result-value">
                                {confidence * 100:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # ----------------------------------
                # 7. VISUAL COMPARISON
                # ----------------------------------

                st.divider()
                st.header("Visual Explanation")

                st.markdown(
                    """
                    Compare the original MRI with the Grad-CAM
                    heatmap and its overlay.
                    """
                )

                image_col1, image_col2, image_col3 = st.columns(3)

                with image_col1:
                    st.subheader("1. Original MRI")

                    st.image(
                        original_image,
                        use_container_width=True
                    )

                    st.caption(
                        "The uploaded brain MRI image."
                    )

                with image_col2:
                    st.subheader("2. Grad-CAM Heatmap")

                    st.image(
                        heatmap_image,
                        use_container_width=True
                    )


                with image_col3:
                    st.subheader("3. Grad-CAM Overlay")

                    st.image(
                        overlay_image,
                        use_container_width=True
                    )

                    st.caption(
                        "The heatmap blended with the original MRI."
                    )


                # ----------------------------------
                # 8. HOW TO INTERPRET THE HEATMAP
                # ----------------------------------

                with st.expander(
                    "How to interpret these visualizations"
                ):
                    st.markdown(
                        """
                        - **Original MRI:** The input image used by the model.
                        - **Heatmap:** Shows the relative strength of
                          Grad-CAM activation across image regions.
                        - **Overlay:** Combines the MRI and heatmap to
                          make the highlighted regions easier to locate.

                        A brighter or warmer-colored region indicates
                        stronger relative activation in this visualization.
                        It does not necessarily correspond to a tumor.

                        Grad-CAM does not produce a tumor boundary,
                        segmentation mask, or medical diagnosis.
                        """
                    )


                # ----------------------------------
                # 9. TROUBLESHOOTING
                # ----------------------------------

                with st.expander(
                    "Visualization appears blank or unclear?"
                ):
                    st.markdown(
                        """
                        **For users:**

                        - Try uploading a clear JPG or PNG MRI image.
                        - Try another image to see whether the issue
                          is specific to the current image.
                        - If the visualization remains unclear,
                          report the issue to the project developer.

                        **For developers:**

                        - Inspect Grad-CAM gradient values.
                        - Verify the target convolutional layer.
                        - Check heatmap normalization.
                        - Confirm that preprocessing matches training.
                        - Check whether the heatmap contains meaningful
                          positive activations before normalization.
                        """
                    )


                # ----------------------------------
                # 10. MEDICAL DISCLAIMER
                # ----------------------------------

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the FastAPI backend. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The analysis request timed out. Please try again."
                )

            except requests.exceptions.HTTPError:
                st.error(
                    f"FastAPI returned an error: {response.text}"
                )

            except Exception as e:
                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )