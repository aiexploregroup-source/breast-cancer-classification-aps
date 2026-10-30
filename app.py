
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import gdown
import os
import io
import pandas as pd
from datetime import datetime

# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BreastScan AI | Intelligent Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "model.h5"

CLASS_NAMES = ["Benign", "Malignant"]

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #10243A;
    --navy2: #172F49;
    --teal: #14B8A6;
    --teal-light: #CCFBF1;
    --bg: #F4F7FB;
    --text: #172B43;
    --muted: #718096;
    --border: #E5EAF1;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: var(--navy);
}

[data-testid="stSidebar"] * {
    color: #E8F1F8;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C2D1DF;
}

[data-testid="stSidebar"] .stRadio label {
    color: white;
    font-weight: 600;
}

.brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
}

.brand span {
    color: #5EEAD4;
}

.sidebar-note {
    font-size: 12px;
    color: #A9C0D3;
    line-height: 1.6;
}

.hero {
    background: linear-gradient(120deg, #10243A, #1A4055);
    border-radius: 22px;
    padding: 38px;
    color: white;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero h1 {
    color: white;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 800;
    letter-spacing: -1.8px;
    margin-bottom: 10px;
}

.hero p {
    color: #CBDCE8;
    font-size: 15px;
    line-height: 1.8;
    max-width: 680px;
}

.eyebrow {
    color: #5EEAD4;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 23px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.6px;
    margin-top: 22px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-top: -8px;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    min-height: 120px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.metric-value {
    color: var(--navy);
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-icon {
    font-size: 23px;
}

.info-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 23px;
    height: 100%;
}

.info-card h3 {
    font-size: 17px;
    color: var(--navy);
    font-weight: 800;
}

.info-card p {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.8;
}

.result-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 26px;
    margin-top: 15px;
}

.result-benign {
    border-top: 5px solid #0D9488;
}

.result-malignant {
    border-top: 5px solid #DC6868;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    color: var(--navy);
}

.confidence {
    font-size: 36px;
    font-weight: 800;
    color: var(--teal);
}

.upload-container {
    background: white;
    border: 1px dashed #9BAFC1;
    border-radius: 18px;
    padding: 20px;
}

.stButton > button {
    background: var(--teal);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 23px;
    font-weight: 800;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #0F9689;
    color: white;
    border: none;
}

.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stProgressBar"] > div {
    background-color: var(--teal);
}

.footer {
    text-align: center;
    padding: 30px 10px;
    color: #8B9AAD;
    font-size: 12px;
    border-top: 1px solid var(--border);
    margin-top: 50px;
}

.disclaimer {
    background: #FFF8E8;
    border: 1px solid #F4D9A2;
    padding: 16px;
    border-radius: 12px;
    color: #74551B;
    font-size: 13px;
    line-height: 1.7;
}

.small-tag {
    display: inline-block;
    background: #DDF7F2;
    color: #087F72;
    border-radius: 30px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL DOWNLOAD AND LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not os.path.exists(MODEL_PATH):

        try:
            result = gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False,
                fuzzy=True
            )

            if result is None or not os.path.exists(MODEL_PATH):
                raise RuntimeError("Google Drive download failed.")

        except Exception as e:
            raise RuntimeError(
                f"Unable to download the model: {e}"
            )

    if os.path.getsize(MODEL_PATH) < 1024:
        raise RuntimeError("Downloaded model file is too small or invalid.")

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")


# =========================================================
# 4. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# 5. LOAD MODEL
# =========================================================

try:
    with st.spinner("Initializing BreastScan AI..."):
        model = load_model()

except Exception as error:
    st.error("Unable to initialize the AI model.")
    st.code(str(error))
    st.info(
        "Check your Google Drive file permissions, model file ID "
        "and TensorFlow compatibility."
    )
    st.stop()


# =========================================================
# 6. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image, model):

    image = image.convert("RGB")

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or 48
    width = input_shape[2] or 48
    channels = input_shape[3] or 3

    if channels == 1:
        image = image.convert("L")
        image = image.resize((width, height))
        image_array = np.array(
            image,
            dtype=np.float32
        )
        image_array = np.expand_dims(image_array, axis=-1)

    else:
        image = image.resize((width, height))
        image_array = np.array(
            image,
            dtype=np.float32
        )

    # IMPORTANT:
    # This normalization must match your model's training pipeline.
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# =========================================================
# 7. PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    processed = preprocess_image(image, model)

    prediction = model.predict(
        processed,
        verbose=0
    )

    prediction = np.asarray(prediction).reshape(-1)

    if len(prediction) == 2:

        # Assumes a two-class softmax output.
        probabilities = prediction.astype(float)

        # Convert logits to probabilities if necessary.
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(np.sum(probabilities), 1.0, atol=0.05)
        ):
            exp_values = np.exp(
                probabilities - np.max(probabilities)
            )
            probabilities = exp_values / np.sum(exp_values)

    elif len(prediction) == 1:

        # Assumes a sigmoid output representing class 1.
        malignant_probability = float(prediction[0])

        if not 0 <= malignant_probability <= 1:
            malignant_probability = 1 / (
                1 + np.exp(-malignant_probability)
            )

        probabilities = np.array([
            1 - malignant_probability,
            malignant_probability
        ])

    else:
        raise ValueError(
            "Unsupported model output. Expected one or two outputs."
        )

    class_index = int(np.argmax(probabilities))

    return class_index, probabilities, processed.shape


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Breast<span>Scan</span> AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sidebar-note">INTELLIGENT IMAGE CLASSIFICATION PLATFORM</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "Choose a section",
        [
            "🏠  Dashboard",
            "🔬  AI Scanner",
            "📊  History",
            "ℹ️  About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<span class="small-tag">● MODEL LOADED</span>',
        unsafe_allow_html=True
    )

    st.write("")

    st.caption("TensorFlow + Keras")
    st.caption("Developed by Team Breast Scan AI")
    st.caption("APS LBS Marg, Lucknow")


# =========================================================
# 9. DASHBOARD
# =========================================================

if page == "🏠  Dashboard":

    st.markdown("""
    <div class="hero">

        <div class="eyebrow">ARTIFICIAL INTELLIGENCE · IMAGE ANALYSIS</div>

        <h1>Intelligent Breast<br>Cell Image Analysis.</h1>

        <p>
        Explore an experimental deep-learning system designed
        to classify uploaded breast cell images into two
        model-defined categories: Benign and Malignant.
        </p>

        <span class="small-tag">CNN POWERED</span>

    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)

    benign_count = sum(
        x["Prediction"] == "Benign"
        for x in st.session_state.history
    )

    malignant_count = sum(
        x["Prediction"] == "Malignant"
        for x in st.session_state.history
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔬</div>
            <div class="metric-label">Images analyzed</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🟢</div>
            <div class="metric-label">Benign predictions</div>
            <div class="metric-value">{benign_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔴</div>
            <div class="metric-label">Malignant predictions</div>
            <div class="metric-value">{malignant_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Explore the platform</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Discover the main features of BreastScan AI.</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 AI Image Scanner</h3>
            <p>Upload a compatible image and obtain
            an experimental CNN classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Prediction Insights</h3>
            <p>Explore the model's output scores
            for the two classification categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="info-card">
            <h3>📁 Session History</h3>
            <p>Review previous predictions made
            during your current session.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="disclaimer">
    <b>⚠️ Research disclaimer:</b> BreastScan AI is an educational
    and experimental machine-learning project. It is not a
    medical device and must not be used for cancer screening,
    diagnosis, treatment decisions or reassurance. Its predictions
    require independent scientific validation.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 10. AI SCANNER
# =========================================================

elif page == "🔬  AI Scanner":

    st.markdown(
        '<div class="section-title">AI Image Scanner</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Upload an image to explore the model output.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="disclaimer">
    <b>Before uploading:</b> This experimental classifier is intended
    for research and educational demonstrations only. A prediction
    cannot determine whether a person has or does not have cancer.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader(
        "Choose a breast cell image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG."
    )

    if uploaded_file:

        try:
            image = Image.open(uploaded_file)
            image.load()

            if image.width < 10 or image.height < 10:
                st.error("Image dimensions are too small.")
                st.stop()

            if image.width * image.height > 25000000:
                st.error("Image is too large to process.")
                st.stop()

            st.markdown("### Uploaded image")

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.image(
                    image,
                    caption="Uploaded image",
                    use_container_width=True
                )

            with col2:

                st.markdown("#### Image information")

                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Width:** {image.width}px")
                st.write(f"**Height:** {image.height}px")
                st.write(f"**Format:** {image.format}")

                st.write("")

                analyze = st.button(
                    "🔬  Analyze Image",
                    use_container_width=True
                )

            if analyze:

                with st.spinner("Running model inference..."):

                    class_index, probabilities, shape = predict_image(
                        image
                    )

                prediction_name = CLASS_NAMES[class_index]
                confidence = float(probabilities[class_index]) * 100

                result = {
                    "Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                    "Filename": uploaded_file.name,
                    "Prediction": prediction_name,
                    "Benign Score": round(
                        float(probabilities[0]) * 100, 2
                    ),
                    "Malignant Score": round(
                        float(probabilities[1]) * 100, 2
                    ),
                    "Model Score": round(confidence, 2)
                }

                st.session_state.history.insert(0, result)
                st.session_state.last_prediction = result

                st.markdown("### Analysis output")

                result_class = (
                    "result-benign"
                    if class_index == 0
                    else "result-malignant"
                )

                st.markdown(f"""
                <div class="result-card {result_class}">
                    <div class="metric-label">MODEL CLASSIFICATION OUTPUT</div>
                    <div class="result-title">{prediction_name}</div>
                    <div class="confidence">{confidence:.2f}%</div>
                    <div class="metric-label">MODEL SCORE FOR SELECTED CLASS</div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                st.markdown("#### Class output scores")

                st.write(
                    f"🟢 **Benign:** {probabilities[0] * 100:.2f}%"
                )

                st.progress(float(probabilities[0]))

                st.write(
                    f"🔴 **Malignant:** {probabilities[1] * 100:.2f}%"
                )

                st.progress(float(probabilities[1]))

                st.caption(
                    f"Processed input tensor: {shape}"
                )

                report = f"""
BREASTSCAN AI
EXPERIMENTAL MODEL OUTPUT REPORT
=================================

Date: {result['Time']}
Filename: {result['Filename']}

Model classification output: {prediction_name}

Benign output score: {probabilities[0] * 100:.2f}%
Malignant output score: {probabilities[1] * 100:.2f}%
Selected class score: {confidence:.2f}%

IMPORTANT MEDICAL DISCLAIMER:
This is an experimental machine-learning output.
It is NOT a medical diagnosis, cancer screening result,
or clinically validated prediction.

A benign prediction does not rule out cancer.
A malignant prediction does not confirm cancer.

Consult a qualified healthcare professional for
medical evaluation and appropriate testing.

Developed by Team Breast Scan AI
APS LBS Marg, Lucknow
"""

                st.download_button(
                    "📥 Download Analysis Report",
                    data=report,
                    file_name="BreastScan_AI_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                st.markdown("""
                <div class="disclaimer">
                <b>Important:</b> The displayed percentages are model
                output scores, not clinically established probabilities
                of having or not having cancer. Even a high score does
                not establish diagnostic certainty.
                </div>
                """, unsafe_allow_html=True)

        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image.")

        except Exception as error:
            st.error(f"Image analysis failed: {error}")


# =========================================================
# 11. HISTORY
# =========================================================

elif page == "📊  History":

    st.markdown(
        '<div class="section-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Results from your current browser session.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Export History as CSV",
            data=csv,
            file_name="breastscan_history.csv",
            mime="text/csv"
        )

        if st.button("🗑 Clear History"):

            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()

    else:

        st.info(
            "No predictions yet. Visit the AI Scanner to analyze an image."
        )


# =========================================================
# 12. ABOUT PROJECT
# =========================================================

elif page == "ℹ️  About Project":

    st.markdown(
        '<div class="section-title">About BreastScan AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">

        <div class="eyebrow">OUR PROJECT</div>

        <h1>Artificial Intelligence<br>for Image Classification.</h1>

        <p>
        BreastScan AI demonstrates how convolutional neural
        networks can be integrated into an interactive
        image-classification application.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project overview")

    st.write("""
    BreastScan AI is an educational machine-learning project
    that uses a custom Convolutional Neural Network (CNN)
    built with TensorFlow and Keras.

    The model produces one of two outputs:

    - Benign
    - Malignant

    Users can upload an image, run model inference and
    explore the classification scores.
    """)

    st.markdown("### Technology stack")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown("**Python**")
        st.caption("Programming")

    with t2:
        st.markdown("**TensorFlow**")
        st.caption("Deep learning")

    with t3:
        st.markdown("**NumPy**")
        st.caption("Numerical processing")

    with t4:
        st.markdown("**Streamlit**")
        st.caption("Web application")

    st.markdown("### Development team")

    st.markdown("""
    **Team Breast Scan AI**

    Army Public School, LBS Marg, Lucknow

    Academic Year: 2026–2027
    """)

    st.markdown(
        "[View GitHub Repository](https://github.com/aiexploregroup-source/breast-cancer-classification-aps)"
    )

    st.markdown("""
    <div class="disclaimer">
    This project is a student research demonstration.
    It has not been established as a clinically validated
    diagnostic tool. Its outputs must not guide medical decisions.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <b>BreastScan AI</b><br>

    Developed by Team Breast Scan AI |
    APS LBS Marg, Lucknow<br>

    © 2026 · Educational AI Research Project

</div>
""", unsafe_allow_html=True)
