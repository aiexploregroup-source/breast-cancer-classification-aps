import os
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
from PIL import Image, UnidentifiedImageError

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    HRFlowable,
    PageBreak,
)

# ============================================================
# 01. CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="BreastScan AI | Intelligent Cell Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_NAME = "BreastScan AI"
FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"
MODEL_PATH = Path("model.h5")
TEMP_MODEL_PATH = Path("model_download.tmp")
CLASS_NAMES = ["Benign", "Malignant"]

# These settings MUST match your training pipeline.
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
NORMALIZATION = "divide_255"

# Choose the output format used by your trained model:
# "softmax" = two-class softmax
# "sigmoid" = one sigmoid output, where 1 means Malignant
MODEL_OUTPUT_MODE = "softmax"

# Change to True only if your model expects one grayscale channel.
GRAYSCALE_MODEL = False
MAX_IMAGE_PIXELS = 25_000_000

# ============================================================
# 02. UIVERSE-INSPIRED DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #07111f;
    --surface: #101e30;
    --surface2: #14263b;
    --surface3: #1b3048;
    --teal: #35dbc5;
    --cyan: #58d8ed;
    --white: #f3f8ff;
    --muted: #9aadc2;
    --border: rgba(148, 180, 210, 0.15);
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(18, 119, 132, .19), transparent 35%),
        radial-gradient(ellipse at 95% 15%, rgba(43, 105, 170, .16), transparent 35%),
        linear-gradient(140deg, #07111f, #0a1727 55%, #081321);
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1c2d, #091522);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #e8f2fc;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted);
}

.sidebar-brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1.2px;
}

.sidebar-brand span {
    color: var(--teal);
}

.sidebar-caption {
    font-size: 10px;
    letter-spacing: 2px;
    color: #91a8bd;
    font-weight: 700;
}

/* ---------- TYPOGRAPHY ---------- */
h1, h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--white) !important;
}

p, label, li {
    color: #c2d0df;
}

.page-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(27px, 4vw, 40px);
    font-weight: 800;
    letter-spacing: -1.7px;
    color: #f3f8ff;
    margin-bottom: 6px;
}

.page-description {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.8;
}

.eyebrow {
    color: var(--teal);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 13px;
}

/* ---------- HERO ---------- */
.hero {
    background:
        radial-gradient(circle at 85% 20%, rgba(53, 219, 197, .19), transparent 30%),
        linear-gradient(120deg, #142c42, #10243a 55%, #13354a);
    border: 1px solid rgba(86, 213, 207, .22);
    border-radius: 25px;
    padding: clamp(25px, 5vw, 48px);
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 60px rgba(0,0,0,.18);
    margin-bottom: 28px;
}

.hero:after {
    content: "✳";
    position: absolute;
    right: 6%;
    top: 5%;
    font-size: 170px;
    color: rgba(75, 224, 204, .06);
    pointer-events: none;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 54px) !important;
    line-height: 1.12;
    letter-spacing: -2.5px;
    max-width: 700px;
}

.hero p {
    color: #bdcfdf;
    max-width: 650px;
    line-height: 1.9;
    font-size: 14px;
}

/* ---------- GLASS CARDS ---------- */
.glass-card {
    background: linear-gradient(
        145deg,
        rgba(25, 47, 69, .88),
        rgba(13, 29, 46, .95)
    );
    border: 1px solid var(--border);
    border-radius: 19px;
    padding: 23px;
    height: 100%;
    box-shadow: 0 10px 30px rgba(0,0,0,.12);
    transition: transform .25s ease, border-color .25s ease,
                box-shadow .25s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    border-color: rgba(53, 219, 197, .5);
    box-shadow: 0 15px 38px rgba(0,0,0,.23);
}

.card-icon {
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: rgba(53, 219, 197, .11);
    font-size: 21px;
    margin-bottom: 17px;
}

.card-title {
    color: #f1f7ff;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 8px;
}

.card-description {
    color: #9eb1c6;
    font-size: 12px;
    line-height: 1.8;
}

/* ---------- METRICS ---------- */
.metric-card {
    background: linear-gradient(145deg, #152a40, #101f31);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 22px;
    transition: .25s ease;
}

.metric-card:hover {
    border-color: rgba(53,219,197,.5);
    transform: translateY(-3px);
}

.metric-label {
    color: #92a8be;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.metric-value {
    color: white;
    font-size: 31px;
    font-weight: 800;
    margin-top: 9px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.metric-icon {
    font-size: 23px;
    margin-bottom: 13px;
}

/* ---------- UIverse-STYLE BUTTONS ---------- */
.stButton > button {
    width: 100%;
    min-height: 45px;
    border: 1px solid rgba(53,219,197,.3);
    border-radius: 12px;
    color: #071923;
    background: linear-gradient(100deg, #35dbc5, #58d8ed);
    font-weight: 800;
    transition: all .25s ease;
    box-shadow: 0 5px 18px rgba(53,219,197,.12);
}

.stButton > button:hover {
    color: #071923;
    border: 1px solid #8fffee;
    box-shadow: 0 0 22px rgba(53,219,197,.35);
    transform: translateY(-2px);
}

.stDownloadButton > button {
    background: #142c42;
    border: 1px solid rgba(53,219,197,.35);
    color: #5eead4;
    border-radius: 11px;
    font-weight: 700;
    min-height: 43px;
}

.stDownloadButton > button:hover {
    color: white;
    border-color: #35dbc5;
}

/* ---------- INPUTS ---------- */
[data-testid="stFileUploader"] {
    background: rgba(20, 39, 58, .8);
    border: 1px dashed rgba(53,219,197,.45);
    border-radius: 16px;
    padding: 16px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] * {
    color: #d9e9f5 !important;
}

.stRadio label {
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- RESULTS ---------- */
.result-card {
    background: linear-gradient(130deg, #142b41, #102237);
    border: 1px solid rgba(53,219,197,.3);
    border-radius: 20px;
    padding: 26px;
    margin: 15px 0;
}

.result-name {
    font-size: 31px;
    font-weight: 800;
    color: white;
    margin: 9px 0;
}

.result-score {
    color: var(--teal);
    font-size: 37px;
    font-weight: 800;
}

.score-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #91a8bd;
    font-weight: 800;
}

.status-pill {
    display: inline-block;
    padding: 6px 12px;
    background: rgba(53,219,197,.12);
    color: #6ef4df;
    border: 1px solid rgba(53,219,197,.3);
    border-radius: 50px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

/* ---------- ALERTS ---------- */
.warning-card {
    background: rgba(151, 104, 29, .13);
    border: 1px solid rgba(234, 179, 8, .35);
    border-radius: 13px;
    padding: 16px;
    color: #f3d89a;
    font-size: 12px;
    line-height: 1.8;
}

.info-card {
    background: rgba(41, 121, 165, .12);
    border: 1px solid rgba(56, 189, 248, .25);
    border-radius: 13px;
    padding: 15px;
    color: #b9e4fa;
    font-size: 12px;
    line-height: 1.8;
}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(14px);}
    to {opacity: 1; transform: translateY(0);}
}

.animate {
    animation: fadeUp .55s ease both;
}

@keyframes pulseGlow {
    0%,100% {box-shadow: 0 0 0 rgba(53,219,197,0);}
    50% {box-shadow: 0 0 20px rgba(53,219,197,.12);}
}

.glow {
    animation: pulseGlow 3s infinite;
}

/* ---------- FOOTER ---------- */
.footer {
    border-top: 1px solid var(--border);
    margin-top: 55px;
    padding: 25px 10px;
    text-align: center;
    color: #71889e;
    font-size: 11px;
    line-height: 2;
}

.footer strong {
    color: #c4d9e8;
}

/* ---------- MOBILE ---------- */
@media(max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 3rem;
    }
    .hero {
        padding: 24px;
    }
    .hero:after {
        font-size: 90px;
    }
    .metric-card {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 03. SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "model_error" not in st.session_state:
    st.session_state.model_error = None

# ============================================================
# 04. MODEL DOWNLOAD
# ============================================================
def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024:
        return

    try:
        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        downloaded = gdown.download(
            url,
            str(TEMP_MODEL_PATH),
            quiet=False,
            fuzzy=True
        )

        if not downloaded or not TEMP_MODEL_PATH.exists():
            raise RuntimeError(
                "Google Drive did not return a valid model file."
            )

        if TEMP_MODEL_PATH.stat().st_size < 1024:
            raise RuntimeError(
                "Downloaded file is too small. Check Drive permissions."
            )

        # Ensure the file is actually a readable HDF5 model.
        with open(TEMP_MODEL_PATH, "rb") as f:
            signature = f.read(8)

        if signature != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError(
                "Downloaded file is not a valid HDF5 model. "
                "Check the Google Drive file ID and sharing permissions."
            )

        os.replace(TEMP_MODEL_PATH, MODEL_PATH)

    finally:
        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()

# ============================================================
# 05. LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def get_model():
    download_model()
    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    if len(loaded_model.inputs) != 1:
        raise ValueError("Only single-input models are supported.")

    shape = loaded_model.input_shape
    if len(shape) != 4:
        raise ValueError(
            f"Expected a 4D image input, received {shape}"
        )

    return loaded_model

with st.spinner("Initializing BreastScan AI engine..."):
    try:
        model = get_model()
        st.session_state.model_error = None
    except Exception as error:
        model = None
        st.session_state.model_error = str(error)

# ============================================================
# 06. IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image):
    if model is None:
        raise RuntimeError("Model is not available.")

    shape = model.input_shape
    if isinstance(shape, list):
        raise ValueError("Multiple model inputs are unsupported.")

    # Detect channels-first vs channels-last.
    channels_first = (
        shape[1] in (1, 3, 4)
        and shape[3] not in (1, 3, 4)
    )

    if channels_first:
        channels = shape[1]
        height = shape[2] or IMAGE_HEIGHT
        width = shape[3] or IMAGE_WIDTH
    else:
        height = shape[1] or IMAGE_HEIGHT
        width = shape[2] or IMAGE_WIDTH
        channels = shape[3] or (1 if GRAYSCALE_MODEL else 3)

    if height is None or width is None:
        height, width = IMAGE_HEIGHT, IMAGE_WIDTH

    height, width = int(height), int(width)
    channels = int(channels)

    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported image channels: {channels}"
        )

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(image, dtype=np.float32)

    if channels == 1:
        array = np.expand_dims(array, axis=-1)

    if NORMALIZATION == "divide_255":
        array = array / 255.0
    elif NORMALIZATION == "minus1_to1":
        array = (array / 127.5) - 1.0
    elif NORMALIZATION == "none":
        pass
    else:
        raise ValueError("Unknown normalization configuration.")

    if channels_first:
        array = np.transpose(array, (2, 0, 1))

    array = np.expand_dims(array, axis=0)
    return array

# ============================================================
# 07. PREDICTION ENGINE
# ============================================================
def get_probabilities(output):
    output = np.asarray(output, dtype=np.float64).reshape(-1)

    if MODEL_OUTPUT_MODE == "softmax":
        if len(output) != 2:
            raise ValueError(
                "Softmax mode requires exactly two outputs."
            )
        # Softmax output must be valid probabilities.
        if (
            np.any(output < 0)
            or np.any(output > 1)
            or not np.isclose(output.sum(), 1.0, atol=0.01)
        ):
            raise ValueError(
                "Output does not resemble a two-class softmax. "
                "Check MODEL_OUTPUT_MODE and the training model."
            )
        probabilities = output

    elif MODEL_OUTPUT_MODE == "sigmoid":
        if len(output) != 1:
            raise ValueError(
                "Sigmoid mode requires exactly one output."
            )
        malignant = float(output[0])
        if not 0 <= malignant <= 1:
            raise ValueError(
                "Sigmoid output is outside [0,1]. "
                "Check whether your model outputs logits."
            )
        probabilities = np.array([
            1.0 - malignant,
            malignant
        ])
    else:
        raise ValueError("Invalid MODEL_OUTPUT_MODE.")

    return probabilities

def analyze_image(image):
    tensor = preprocess_image(image)
    prediction = model.predict(
        tensor,
        verbose=0
    )
    probabilities = get_probabilities(prediction)
    index = int(np.argmax(probabilities))
    return index, probabilities, tensor.shape

# ============================================================
# 07b. PROFESSIONAL PDF REPORT GENERATOR
# ============================================================
def generate_analysis_pdf(result, original_image, probabilities, tensor_shape, model):
    """
    Generate a clean two-page BreastScan AI experimental analysis report.
    """
    buffer = BytesIO()

    navy = colors.HexColor("#10243A")
    dark = colors.HexColor("#172F49")
    teal = colors.HexColor("#0D9488")
    light_teal = colors.HexColor("#E8F8F5")
    light_bg = colors.HexColor("#F3F6FA")
    grey = colors.HexColor("#65758B")
    dark_text = colors.HexColor("#24364B")
    border = colors.HexColor("#DCE5ED")
    amber = colors.HexColor("#FFF5DF")
    amber_text = colors.HexColor("#805B18")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Brand", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=25, leading=30,
        textColor=navy, alignment=TA_LEFT, spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="SubtitleCustom", parent=styles["Normal"],
        fontSize=9, leading=14, textColor=grey
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=17,
        textColor=navy, spaceBefore=12, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"],
        fontSize=9, leading=14, textColor=dark_text
    ))
    styles.add(ParagraphStyle(
        name="SmallCustom", parent=styles["Normal"],
        fontSize=7.5, leading=11, textColor=grey
    ))
    styles.add(ParagraphStyle(
        name="WhiteSmall", parent=styles["Normal"],
        fontSize=8, leading=12, textColor=colors.HexColor("#D6E7F1")
    ))
    styles.add(ParagraphStyle(
        name="DisclaimerCustom", parent=styles["Normal"],
        fontSize=8, leading=12, textColor=amber_text
    ))
    styles.add(ParagraphStyle(
        name="CenteredSmall", parent=styles["Normal"],
        fontSize=8, leading=12, textColor=grey, alignment=TA_CENTER
    ))

    def page_decor(canvas, doc):
        canvas.saveState()
        page_width, page_height = A4
        canvas.setFillColor(teal)
        canvas.rect(0, page_height - 5, page_width, 5, stroke=0, fill=1)
        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 15 * mm, page_width - 18 * mm, 15 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(grey)
        canvas.drawString(18 * mm, 10 * mm, "BREASTSCAN AI | EDUCATIONAL RESEARCH PROJECT")
        canvas.drawRightString(page_width - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=22 * mm,
        title="BreastScan AI Analysis Report",
        author="Team Breast Scan AI",
        subject="Experimental CNN image classification report",
    )

    story = []

    # ---------- HEADER ----------
    story.append(Paragraph("BREASTSCAN <font color='#0D9488'>AI</font>", styles["Brand"]))
    story.append(Paragraph("INTELLIGENT IMAGE CLASSIFICATION PLATFORM", styles["SmallCustom"]))
    story.append(Spacer(1, 7))
    story.append(HRFlowable(width="100%", thickness=1.2, color=teal, spaceAfter=12))
    story.append(Paragraph("EXPERIMENTAL IMAGE ANALYSIS REPORT", styles["SectionCustom"]))
    story.append(Paragraph(
        "A machine-learning inference report generated by the BreastScan AI educational application.",
        styles["SubtitleCustom"]
    ))
    story.append(Spacer(1, 13))

    # ---------- RESULT BANNER ----------
    prediction = escape(str(result["Prediction"]))
    score = float(result["Selected Class Score"])
    result_data = [[
        Paragraph(
            "<font color='#D6E7F1'>MODEL CLASSIFICATION OUTPUT</font><br/>"
            f"<font color='#FFFFFF' size='21'><b>{prediction}</b></font>",
            styles["WhiteSmall"]
        ),
        Paragraph(
            f"<font color='#5EEAD4' size='23'><b>{score:.2f}%</b></font><br/>"
            "<font color='#D6E7F1'>SELECTED CLASS SCORE</font>",
            styles["CenteredSmall"]
        )
    ]]
    result_table = Table(result_data, colWidths=[88 * mm, 75 * mm], rowHeights=[29 * mm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), navy),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 13),
        ("RIGHTPADDING", (0, 0), (-1, -1), 13),
        ("BOX", (0, 0), (-1, -1), 0.7, teal),
        ("ROUNDEDCORNERS", [9, 9, 9, 9]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 13))

    # ---------- IMAGE + DETAILS ----------
    story.append(Paragraph("01 | IMAGE INFORMATION", styles["SectionCustom"]))

    image_buffer = BytesIO()
    preview = original_image.convert("RGB")
    preview.thumbnail((900, 700))
    preview.save(image_buffer, format="JPEG", quality=88)
    image_buffer.seek(0)
    preview_image = PDFImage(image_buffer, width=67 * mm, height=51 * mm, kind="proportional")

    image_details = [
        ["Filename", escape(str(result["Filename"]))],
        ["Analysis date", escape(str(result["Time"]))],
        ["Original dimensions", f"{original_image.width} × {original_image.height} pixels"],
        ["Model input tensor", escape(str(tensor_shape))],
        ["Model architecture", escape(str(model.name))],
    ]
    detail_table = Table(image_details, colWidths=[36 * mm, 88 * mm])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), light_bg),
        ("TEXTCOLOR", (0, 0), (0, -1), navy),
        ("TEXTCOLOR", (1, 0), (1, -1), dark_text),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.4, border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    image_layout = Table([[preview_image, detail_table]], colWidths=[70 * mm, 94 * mm])
    image_layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(image_layout)

    # ---------- SCORES ----------
    story.append(Paragraph("02 | MODEL OUTPUT SCORES", styles["SectionCustom"]))
    benign_score = float(probabilities[0]) * 100
    malignant_score = float(probabilities[1]) * 100
    score_data = [
        ["CLASS", "MODEL OUTPUT SCORE"],
        ["Benign", f"{benign_score:.2f}%"],
        ["Malignant", f"{malignant_score:.2f}%"],
    ]
    score_table = Table(score_data, colWidths=[81 * mm, 82 * mm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), light_teal),
        ("BACKGROUND", (0, 2), (-1, 2), light_bg),
        ("TEXTCOLOR", (0, 1), (-1, -1), dark_text),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        "These values represent the model's classification output scores. "
        "They are not calibrated probabilities of disease.",
        styles["SmallCustom"]
    ))

    # ---------- PAGE 2 ----------
    story.append(PageBreak())
    story.append(Paragraph("TECHNICAL NOTES & LIMITATIONS", styles["Brand"]))
    story.append(Paragraph("Supporting information for this experimental analysis.", styles["SubtitleCustom"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=teal, spaceBefore=10, spaceAfter=12))

    story.append(Paragraph("03 | MODEL INFORMATION", styles["SectionCustom"]))
    model_details = [
        ["Property", "Value"],
        ["Application", "BreastScan AI"],
        ["Model framework", "TensorFlow / Keras"],
        ["Model architecture", escape(str(model.name))],
        ["Model input", escape(str(tensor_shape))],
        ["Classification classes", "Benign / Malignant"],
    ]
    model_table = Table(model_details, colWidths=[64 * mm, 99 * mm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), light_bg),
        ("TEXTCOLOR", (0, 1), (-1, -1), dark_text),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(model_table)

    story.append(Paragraph("04 | INTERPRETATION", styles["SectionCustom"]))
    story.append(Paragraph(
        "The application processes an uploaded image and passes the resulting image tensor "
        "to a trained convolutional neural network. The network returns scores corresponding "
        "to its configured output classes. The displayed category is the class with the "
        "highest model output score.",
        styles["BodyCustom"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("05 | IMPORTANT MEDICAL AND RESEARCH DISCLAIMER", styles["SectionCustom"]))
    disclaimer_text = """
    <b>THIS IS NOT A MEDICAL DIAGNOSTIC REPORT.</b><br/><br/>
    BreastScan AI is an educational and experimental machine-learning project.
    It has not been established as a clinically validated medical device or cancer screening system.<br/><br/>
    A Benign classification does not rule out cancer. A Malignant classification does not confirm cancer.
    Model output scores should not be interpreted as a person's probability of having cancer.<br/><br/>
    Performance on an individual image may be affected by image quality, dataset characteristics,
    preprocessing, model limitations and differences between training and real-world images.<br/><br/>
    Do not use this application or its reports for diagnosis, screening, treatment selection or medical decisions.
    Medical concerns must be evaluated by a qualified healthcare professional using appropriate clinical investigations.
    """
    disclaimer_box = Table(
        [[Paragraph(disclaimer_text, styles["DisclaimerCustom"])]],
        colWidths=[163 * mm]
    )
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), amber),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#E8C77B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(disclaimer_box)

    story.append(Paragraph("06 | PROJECT ACKNOWLEDGEMENT", styles["SectionCustom"]))
    story.append(Paragraph(
        "<b>Developed by Team Breast Scan AI</b><br/>"
        "Army Public School, LBS Marg, Lucknow<br/>"
        "Academic Year 2026–2027",
        styles["BodyCustom"]
    ))
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.8, color=border))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "BREASTSCAN AI · EXPERIMENTAL RESEARCH REPORT",
        styles["CenteredSmall"]
    ))
    story.append(Paragraph(
        "Generated automatically by the BreastScan AI application.",
        styles["CenteredSmall"]
    ))

    # Build the PDF
    doc.build(
        story,
        onFirstPage=page_decor,
        onLaterPages=page_decor
    )
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# 08. REUSABLE UI COMPONENTS
# ============================================================
def page_header(title, description, eyebrow="BREASTSCAN AI"):
    st.markdown(f"""
    <div class="animate">
        <div class="eyebrow">{eyebrow}</div>
        <div class="page-heading">{title}</div>
        <div class="page-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(icon, label, value):
    st.markdown(f"""
    <div class="metric-card animate">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def info_card(icon, title, description):
    st.markdown(f"""
    <div class="glass-card animate">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def disclaimer():
    st.markdown("""
    <div class="warning-card">
        <strong>⚠ RESEARCH USE ONLY</strong><br>
        BreastScan AI is an educational image-classification
        demonstration, not a clinically validated diagnostic
        or screening system. A Benign output cannot rule out
        cancer, and a Malignant output cannot confirm cancer.
        Never use these results to make medical decisions.
        Consult a qualified healthcare professional for medical
        assessment.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 09. SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">Breast<span>Scan</span> AI</div>
    <div class="sidebar-caption">INTELLIGENT IMAGE ANALYSIS</div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("---")
    st.markdown("##### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "AI Scanner",
            "Prediction History",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if model is not None:
        st.markdown(
            '<span class="status-pill">● MODEL ONLINE</span>',
            unsafe_allow_html=True
        )
    else:
        st.error("Model unavailable")

    st.write("")
    st.caption("POWERED BY TENSORFLOW")
    st.caption("TEAM BREAST SCAN AI")
    st.caption("APS LBS MARG · LUCKNOW")
    st.markdown("---")
    st.caption("ACADEMIC PROJECT · 2026")

# ============================================================
# 10. MODEL ERROR SCREEN
# ============================================================
if model is None:
    st.error("AI model could not be initialized.")
    st.write(
        "Check the Google Drive link, model format, "
        "file permissions and TensorFlow compatibility."
    )
    st.code(st.session_state.model_error or "Unknown error")
    st.stop()

# ============================================================
# 11. DASHBOARD
# ============================================================
if page == "Dashboard":
    st.markdown("""
    <div class="hero animate">
        <div class="eyebrow">ARTIFICIAL INTELLIGENCE · DEEP LEARNING</div>
        <h1>Explore the power<br>of intelligent image analysis.</h1>
        <p>
        An educational demonstration of convolutional neural
        networks for breast cell image classification.
        Explore model outputs through an interactive workspace.
        </p>
        <span class="status-pill">CNN POWERED</span>
        <span class="status-pill">TENSORFLOW</span>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    benign = sum(
        x["Prediction"] == "Benign"
        for x in st.session_state.history
    )
    malignant = sum(
        x["Prediction"] == "Malignant"
        for x in st.session_state.history
    )

    a, b, c = st.columns(3)
    with a:
        metric_card("🔬", "Images analyzed", total)
    with b:
        metric_card("🟢", "Benign outputs", benign)
    with c:
        metric_card("🔴", "Malignant outputs", malignant)

    st.write("")
    page_header(
        "Your workspace",
        "Everything you need to explore the experimental model."
    )

    x, y, z = st.columns(3)
    with x:
        info_card(
            "🔬",
            "AI Scanner",
            "Upload a compatible image and run model inference."
        )
    with y:
        info_card(
            "📊",
            "Output Analysis",
            "Inspect both classification scores and the selected class."
        )
    with z:
        info_card(
            "📁",
            "Session History",
            "Review and export your current session's predictions."
        )

    st.write("")
    disclaimer()

# ============================================================
# 12. AI SCANNER
# ============================================================
elif page == "AI Scanner":
    page_header(
        "AI Image Scanner",
        "Upload an image to explore your model's classification output.",
        "IMAGE ANALYSIS LAB"
    )

    st.write("")
    disclaimer()
    st.write("")

    uploaded_file = st.file_uploader(
        "UPLOAD IMAGE",
        type=["jpg", "jpeg", "png"],
        help="Choose a JPG or PNG image."
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()

            if len(image_bytes) > 15 * 1024 * 1024:
                st.error("File exceeds the 15 MB upload limit.")
                st.stop()

            file_hash = hashlib.sha256(
                image_bytes
            ).hexdigest()

            image = Image.open(
                uploaded_file
            )
            image.load()

            if image.width * image.height > MAX_IMAGE_PIXELS:
                st.error("Image resolution is too large.")
                st.stop()

            if image.width < 10 or image.height < 10:
                st.error("Image resolution is too small.")
                st.stop()

            col1, col2 = st.columns([1.15, 1])

            with col1:
                st.markdown("### Image preview")
                st.image(
                    image,
                    use_container_width=True
                )

            with col2:
                st.markdown("### Image details")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Dimensions:** {image.width} × {image.height}")
                st.write(f"**Format:** {image.format}")
                st.write(f"**File size:** {len(image_bytes) / 1024:.1f} KB")

                st.write("")
                analyze = st.button(
                    "✦  RUN AI ANALYSIS",
                    key="analyze_button",
                    use_container_width=True
                )

            if analyze:
                with st.spinner("Analyzing image with CNN..."):
                    index, probabilities, tensor_shape = analyze_image(image)

                prediction_name = CLASS_NAMES[index]
                selected_score = float(
                    probabilities[index] * 100
                )

                result = {
                    "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "Filename": uploaded_file.name,
                    "Prediction": prediction_name,
                    "Benign Score": round(
                        float(probabilities[0] * 100), 2
                    ),
                    "Malignant Score": round(
                        float(probabilities[1] * 100), 2
                    ),
                    "Selected Class Score": round(selected_score, 2),
                    "File Hash": file_hash
                }

                st.session_state.history.insert(0, result)
                st.session_state.last_prediction = result

                st.write("")
                st.markdown("### Analysis results")

                st.markdown(f"""
                <div class="result-card glow animate">
                    <div class="score-label">MODEL CLASSIFICATION OUTPUT</div>
                    <div class="result-name">{prediction_name}</div>
                    <div class="result-score">{selected_score:.2f}%</div>
                    <div class="score-label">SELECTED CLASS OUTPUT SCORE</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Class output scores")
                p1, p2 = st.columns(2)

                with p1:
                    st.markdown("**🟢 Benign**")
                    st.metric(
                        "Model score",
                        f"{probabilities[0] * 100:.2f}%"
                    )
                    st.progress(
                        float(probabilities[0])
                    )

                with p2:
                    st.markdown("**🔴 Malignant**")
                    st.metric(
                        "Model score",
                        f"{probabilities[1] * 100:.2f}%"
                    )
                    st.progress(
                        float(probabilities[1])
                    )

                st.caption(
                    f"Model input tensor: {tensor_shape}"
                )

                # Generate professional PDF report
                pdf_report = generate_analysis_pdf(
                    result=result,
                    original_image=image,
                    probabilities=probabilities,
                    tensor_shape=tensor_shape,
                    model=model
                )

                st.download_button(
                    label="📄 DOWNLOAD PROFESSIONAL PDF REPORT",
                    data=pdf_report,
                    file_name=(
                        f"BreastScan_AI_Report_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_report"
                )

                disclaimer()

        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload a valid JPG or PNG.")
        except Exception as error:
            st.error("Image analysis failed.")
            st.code(str(error))

# ============================================================
# 13. PREDICTION HISTORY
# ============================================================
elif page == "Prediction History":
    page_header(
        "Prediction History",
        "Review and export predictions made in your current session.",
        "SESSION ANALYTICS"
    )

    st.write("")

    if st.session_state.history:
        df = pd.DataFrame(
            st.session_state.history
        )
        display_df = df.drop(
            columns=["File Hash"],
            errors="ignore"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        a, b = st.columns(2)
        with a:
            st.download_button(
                "↓ EXPORT CSV",
                data=csv_data,
                file_name="BreastScan_History.csv",
                mime="text/csv",
                use_container_width=True
            )
        with b:
            if st.button(
                "CLEAR HISTORY",
                use_container_width=True
            ):
                st.session_state.history = []
                st.session_state.last_prediction = None
                st.rerun()
    else:
        st.info(
            "Your history is empty. Analyze an image to see results here."
        )

# ============================================================
# 14. ABOUT PROJECT
# ============================================================
elif page == "About Project":
    page_header(
        "About BreastScan AI",
        "An educational deep-learning and computer vision project.",
        "PROJECT INFORMATION"
    )

    st.write("")
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">TEAM BREAST SCAN AI</div>
        <h1>Technology meets<br>medical image research.</h1>
        <p>
        This project demonstrates the integration of a
        Convolutional Neural Network with an interactive
        image-classification web application.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        info_card(
            "🧠",
            "Deep Learning",
            "A custom CNN model developed using TensorFlow and Keras."
        )
    with b:
        info_card(
            "🌐",
            "Web Technology",
            "An interactive application developed using Streamlit."
        )

    st.write("")
    st.markdown("### Technology stack")

    tech = pd.DataFrame({
        "Technology": [
            "Python",
            "TensorFlow",
            "Keras",
            "NumPy",
            "Pillow",
            "Streamlit",
            "ReportLab"
        ],
        "Purpose": [
            "Programming",
            "Deep learning",
            "Neural network framework",
            "Numerical operations",
            "Image processing",
            "Web application",
            "Professional PDF reports"
        ]
    })

    st.dataframe(
        tech,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Model configuration")
    st.write(f"**Input dimensions:** {IMAGE_WIDTH} × {IMAGE_HEIGHT}")
    st.write(f"**Normalization:** {NORMALIZATION}")
    st.write(f"**Output mode:** {MODEL_OUTPUT_MODE}")
    st.write(f"**Model architecture name:** {model.name}")

    st.write("")
    st.markdown("### Development team")
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">Team Breast Scan AI</div>
        <div class="card-description">
        Army Public School, LBS Marg, Lucknow<br>
        Academic Year 2026–2027
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown(
        "[VIEW GITHUB REPOSITORY](https://github.com/aiexploregroup-source/breast-cancer-classification-aps)"
    )

    disclaimer()

# ============================================================
# 15. FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>BreastScan AI</strong><br>
    An Educational Artificial Intelligence Project<br>
    Developed by Team Breast Scan AI · APS LBS Marg, Lucknow<br>
    © 2026 · All Rights Reserved
</div>
""", unsafe_allow_html=True)
