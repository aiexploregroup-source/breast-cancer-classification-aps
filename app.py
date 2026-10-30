import os
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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as PDFImage, HRFlowable, PageBreak
)

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="BreastScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"
MODEL_PATH = Path("model.h5")
TEMP_MODEL_PATH = Path("model_download.tmp")
CLASS_NAMES = ["Benign", "Malignant"]
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
NORMALIZATION = "divide_255"
MODEL_OUTPUT_MODE = "softmax"
GRAYSCALE_MODEL = False
MAX_IMAGE_PIXELS = 25_000_000

# ============================================================
# CLEAN DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0b1120 !important;
}

.stApp {
    background: 
        linear-gradient(rgba(8, 12, 24, 0.82), rgba(8, 12, 24, 0.88)),
        url("bg.png") center/cover fixed no-repeat !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1100px;
    padding: 1.5rem 1.5rem 3rem;
}

/* Hide default sidebar completely for cleaner look */
[data-testid="stSidebar"] { display: none; }

/* ---------- LANDING ---------- */
.landing {
    min-height: 82vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem 1rem;
}

.logo-wrap {
    margin-bottom: 1.8rem;
}

.logo-wrap img {
    width: 92px;
    height: 92px;
    object-fit: contain;
    filter: drop-shadow(0 8px 24px rgba(45, 212, 191, 0.25));
}

.landing h1 {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #f8fafc;
    margin: 0 0 0.6rem;
    line-height: 1.15;
}

.landing h1 span {
    background: linear-gradient(135deg, #2dd4bf, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.landing-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 2.2rem;
}

.cta-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    justify-content: center;
}

/* Primary CTA button styling via Streamlit */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
    color: #042f2e !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    min-height: 52px !important;
    box-shadow: 0 8px 24px rgba(20, 184, 166, 0.3) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(20, 184, 166, 0.4) !important;
}

/* Secondary / ghost buttons */
div[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ---------- TOP NAV ---------- */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0 1.4rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
}

.topnav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: #f1f5f9;
}

.topnav-brand img {
    width: 36px;
    height: 36px;
    object-fit: contain;
}

.topnav-links {
    display: flex;
    gap: 6px;
}

/* ---------- CARDS ---------- */
.card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
}

.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.4rem;
}

.card-desc {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.55;
}

/* ---------- RESULT ---------- */
.result-box {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.75));
    border: 1px solid rgba(45, 212, 191, 0.3);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1.2rem 0;
}

.result-label {
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
}

.result-class {
    font-family: 'Outfit', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.3rem 0;
}

.result-score {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #2dd4bf;
}

/* ---------- ALERT ---------- */
.alert {
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #fcd34d;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    color: #64748b;
    font-size: 0.8rem;
    line-height: 1.8;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.6);
    border: 1.5px dashed rgba(45, 212, 191, 0.35);
    border-radius: 14px;
    padding: 1rem;
}

[data-testid="stFileUploader"] * {
    color: #cbd5e1 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 1rem;
}

h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: #f1f5f9 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "history" not in st.session_state:
    st.session_state.history = []
if "model_error" not in st.session_state:
    st.session_state.model_error = None

# ============================================================
# MODEL LOGIC (unchanged)
# ============================================================
def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024:
        return
    try:
        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        downloaded = gdown.download(url, str(TEMP_MODEL_PATH), quiet=False, fuzzy=True)
        if not downloaded or not TEMP_MODEL_PATH.exists():
            raise RuntimeError("Google Drive did not return a valid model file.")
        if TEMP_MODEL_PATH.stat().st_size < 1024:
            raise RuntimeError("Downloaded file is too small.")
        with open(TEMP_MODEL_PATH, "rb") as f:
            if f.read(8) != b"\x89HDF\r\n\x1a\n":
                raise RuntimeError("Downloaded file is not a valid HDF5 model.")
        os.replace(TEMP_MODEL_PATH, MODEL_PATH)
    finally:
        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()

@st.cache_resource(show_spinner=False)
def get_model():
    download_model()
    m = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if len(m.inputs) != 1 or len(m.input_shape) != 4:
        raise ValueError("Model input shape not supported.")
    return m

with st.spinner("Loading model..."):
    try:
        model = get_model()
        st.session_state.model_error = None
    except Exception as e:
        model = None
        st.session_state.model_error = str(e)

def preprocess_image(image):
    shape = model.input_shape
    channels_first = shape[1] in (1, 3, 4) and shape[3] not in (1, 3, 4)
    if channels_first:
        channels, height, width = shape[1], shape[2] or IMAGE_HEIGHT, shape[3] or IMAGE_WIDTH
    else:
        height, width = shape[1] or IMAGE_HEIGHT, shape[2] or IMAGE_WIDTH
        channels = shape[3] or (1 if GRAYSCALE_MODEL else 3)
    height, width, channels = int(height), int(width), int(channels)
    image = image.convert("L" if channels == 1 else "RGB")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    if channels == 1:
        arr = np.expand_dims(arr, -1)
    if NORMALIZATION == "divide_255":
        arr /= 255.0
    elif NORMALIZATION == "minus1_to1":
        arr = (arr / 127.5) - 1.0
    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)

def get_probabilities(output):
    output = np.asarray(output, dtype=np.float64).reshape(-1)
    if MODEL_OUTPUT_MODE == "softmax":
        if len(output) != 2:
            raise ValueError("Softmax needs 2 outputs")
        return output
    elif MODEL_OUTPUT_MODE == "sigmoid":
        m = float(output[0])
        return np.array([1.0 - m, m])
    raise ValueError("Invalid MODEL_OUTPUT_MODE")

def analyze_image(image):
    tensor = preprocess_image(image)
    pred = model.predict(tensor, verbose=0)
    probs = get_probabilities(pred)
    return int(np.argmax(probs)), probs, tensor.shape

# ============================================================
# PDF GENERATOR (same logic)
# ============================================================
def generate_analysis_pdf(result, original_image, probabilities, tensor_shape, model):
    buffer = BytesIO()
    navy = colors.HexColor("#0f172a")
    teal = colors.HexColor("#0d9488")
    light_teal = colors.HexColor("#ecfdf5")
    light_bg = colors.HexColor("#f1f5f9")
    grey = colors.HexColor("#64748b")
    dark_text = colors.HexColor("#1e293b")
    border = colors.HexColor("#e2e8f0")
    amber = colors.HexColor("#fffbeb")
    amber_text = colors.HexColor("#92400e")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=22, leading=27, textColor=navy, spaceAfter=4))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=9, leading=13, textColor=grey))
    styles.add(ParagraphStyle(name="Sec", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=12, leading=16, textColor=navy, spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9, leading=13, textColor=dark_text))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=7.5, leading=11, textColor=grey))
    styles.add(ParagraphStyle(name="White", parent=styles["Normal"], fontSize=8, leading=12,
                              textColor=colors.HexColor("#e0f2fe")))
    styles.add(ParagraphStyle(name="Disc", parent=styles["Normal"], fontSize=8, leading=12, textColor=amber_text))
    styles.add(ParagraphStyle(name="Center", parent=styles["Normal"], fontSize=8, leading=12,
                              textColor=grey, alignment=TA_CENTER))

    def decor(c, doc):
        c.saveState()
        pw, ph = A4
        c.setFillColor(teal)
        c.rect(0, ph - 4, pw, 4, stroke=0, fill=1)
        c.setStrokeColor(border)
        c.setLineWidth(0.5)
        c.line(18*mm, 14*mm, pw-18*mm, 14*mm)
        c.setFont("Helvetica", 7)
        c.setFillColor(grey)
        c.drawString(18*mm, 9*mm, "BREASTSCAN AI  ·  EDUCATIONAL PROJECT")
        c.drawRightString(pw-18*mm, 9*mm, f"Page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=20*mm,
                            title="BreastScan AI Report", author="Team Breast Scan AI")

    story = []
    story.append(Paragraph("BREASTSCAN <font color='#0d9488'>AI</font>", styles["Brand"]))
    story.append(Paragraph("INTELLIGENT IMAGE CLASSIFICATION", styles["Small"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=teal, spaceAfter=10))
    story.append(Paragraph("EXPERIMENTAL ANALYSIS REPORT", styles["Sec"]))
    story.append(Paragraph("Machine-learning inference report from BreastScan AI.", styles["Sub"]))
    story.append(Spacer(1, 10))

    pred = escape(str(result["Prediction"]))
    score = float(result["Selected Class Score"])
    data = [[
        Paragraph(f"<font color='#bae6fd'>MODEL OUTPUT</font><br/><font color='white' size='18'><b>{pred}</b></font>", styles["White"]),
        Paragraph(f"<font color='#5eead4' size='20'><b>{score:.2f}%</b></font><br/><font color='#bae6fd'>SCORE</font>", styles["Center"])
    ]]
    t = Table(data, colWidths=[90*mm, 72*mm], rowHeights=[26*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), navy),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("BOX", (0,0), (-1,-1), 0.8, teal),
        ("ROUNDEDCORNERS", [8,8,8,8]),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("01  ·  IMAGE INFORMATION", styles["Sec"]))
    buf = BytesIO()
    preview = original_image.convert("RGB")
    preview.thumbnail((850, 650))
    preview.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    img = PDFImage(buf, width=64*mm, height=48*mm, kind="proportional")

    details = [
        ["Filename", escape(str(result["Filename"]))],
        ["Date", escape(str(result["Time"]))],
        ["Dimensions", f"{original_image.width} × {original_image.height} px"],
        ["Input tensor", escape(str(tensor_shape))],
        ["Architecture", escape(str(model.name))],
    ]
    dt = Table(details, colWidths=[36*mm, 90*mm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), light_bg),
        ("TEXTCOLOR", (0,0), (0,-1), navy),
        ("TEXTCOLOR", (1,0), (1,-1), dark_text),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.4, border),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(Table([[img, dt]], colWidths=[68*mm, 96*mm]))

    story.append(Paragraph("02  ·  OUTPUT SCORES", styles["Sec"]))
    scores = [
        ["CLASS", "SCORE"],
        ["Benign", f"{float(probabilities[0])*100:.2f}%"],
        ["Malignant", f"{float(probabilities[1])*100:.2f}%"],
    ]
    stbl = Table(scores, colWidths=[80*mm, 82*mm])
    stbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), navy),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,1), light_teal),
        ("BACKGROUND", (0,2), (-1,2), light_bg),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("GRID", (0,0), (-1,-1), 0.5, border),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(stbl)
    story.append(Spacer(1, 5))
    story.append(Paragraph("These are model output scores, not calibrated medical probabilities.", styles["Small"]))

    story.append(PageBreak())
    story.append(Paragraph("TECHNICAL NOTES & DISCLAIMER", styles["Brand"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=teal, spaceBefore=6, spaceAfter=10))

    story.append(Paragraph("03  ·  MODEL INFO", styles["Sec"]))
    minfo = [
        ["Property", "Value"],
        ["Application", "BreastScan AI"],
        ["Framework", "TensorFlow / Keras"],
        ["Architecture", escape(str(model.name))],
        ["Input", escape(str(tensor_shape))],
        ["Classes", "Benign / Malignant"],
    ]
    mt = Table(minfo, colWidths=[60*mm, 102*mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), navy),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,-1), light_bg),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.4, border),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(mt)

    story.append(Paragraph("04  ·  IMPORTANT DISCLAIMER", styles["Sec"]))
    disc = """
    <b>THIS IS NOT A MEDICAL DIAGNOSTIC REPORT.</b><br/><br/>
    BreastScan AI is an educational project only. It is not a clinically validated device.<br/><br/>
    A Benign result does not rule out cancer. A Malignant result does not confirm cancer.<br/><br/>
    Do not use these results for diagnosis, screening or treatment decisions.
    Always consult a qualified healthcare professional.
    """
    db = Table([[Paragraph(disc, styles["Disc"])]], colWidths=[162*mm])
    db.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), amber),
        ("BOX", (0,0), (-1,-1), 0.8, colors.HexColor("#f59e0b")),
        ("LEFTPADDING", (0,0), (-1,-1), 11),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(db)

    story.append(Paragraph("05  ·  TEAM", styles["Sec"]))
    story.append(Paragraph(
        "<b>Team Breast Scan AI</b><br/>Army Public School, LBS Marg, Lucknow<br/>Academic Year 2026–2027",
        styles["Body"]))
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.6, color=border))
    story.append(Spacer(1, 6))
    story.append(Paragraph("BREASTSCAN AI  ·  EDUCATIONAL RESEARCH REPORT", styles["Center"]))

    doc.build(story, onFirstPage=decor, onLaterPages=decor)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# HELPERS
# ============================================================
def go(page):
    st.session_state.page = page
    st.rerun()

def top_nav():
    st.markdown(f"""
    <div class="topnav">
        <div class="topnav-brand">
            <img src="logo.png" alt="logo">
            BreastScan AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 4])
    with cols[0]:
        if st.button("Home", key="nav_home", use_container_width=True):
            go("landing")
    with cols[1]:
        if st.button("Scanner", key="nav_scan", use_container_width=True):
            go("scanner")
    with cols[2]:
        if st.button("History", key="nav_hist", use_container_width=True):
            go("history")
    with cols[3]:
        if st.button("About", key="nav_about", use_container_width=True):
            go("about")

def disclaimer():
    st.markdown("""
    <div class="alert">
        <strong>Research use only.</strong>  
        This is an educational demonstration. Results are not medical diagnoses. 
        Always consult a qualified doctor.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL ERROR
# ============================================================
if model is None:
    st.error("Model could not be loaded.")
    st.code(st.session_state.model_error or "Unknown error")
    st.stop()

# ============================================================
# LANDING PAGE
# ============================================================
if st.session_state.page == "landing":
    st.markdown("""
    <div class="landing">
        <div class="logo-wrap">
            <img src="logo.png" alt="BreastScan AI">
        </div>
        <h1>Intelligent analysis of<br><span>breast cell imagery</span></h1>
        <p class="landing-sub">
            An educational deep-learning tool that classifies cell images 
            using a convolutional neural network. Built for learning and research.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons
    c1, c2, c3 = st.columns([1.4, 1.2, 1.4])
    with c2:
        if st.button("Start Analysis →", key="cta_start", use_container_width=True):
            go("scanner")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="card">
            <div class="card-title">CNN Powered</div>
            <div class="card-desc">Custom convolutional neural network trained for binary cell classification.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Instant Results</div>
            <div class="card-desc">Upload an image and receive classification scores in seconds.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="card">
            <div class="card-title">PDF Reports</div>
            <div class="card-desc">Download a clean professional report of every analysis.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Team Breast Scan AI · Army Public School, LBS Marg, Lucknow<br>
        Academic Project 2026 · For educational purposes only
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SCANNER
# ============================================================
elif st.session_state.page == "scanner":
    top_nav()
    st.markdown("### AI Image Scanner")
    st.caption("Upload a JPG or PNG image to run the model.")
    st.write("")
    disclaimer()
    st.write("")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            data = uploaded.getvalue()
            if len(data) > 15 * 1024 * 1024:
                st.error("File too large (max 15 MB).")
                st.stop()

            image = Image.open(uploaded)
            image.load()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                st.error("Image resolution too high.")
                st.stop()

            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.image(image, use_container_width=True)
            with col2:
                st.markdown("**Image details**")
                st.write(f"Name: `{uploaded.name}`")
                st.write(f"Size: {image.width} × {image.height}")
                st.write(f"Format: {image.format}")
                st.write(f"File: {len(data)/1024:.1f} KB")
                st.write("")
                run = st.button("Run Analysis", use_container_width=True)

            if run:
                with st.spinner("Analyzing..."):
                    idx, probs, shape = analyze_image(image)

                name = CLASS_NAMES[idx]
                score = float(probs[idx] * 100)

                result = {
                    "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "Filename": uploaded.name,
                    "Prediction": name,
                    "Benign Score": round(float(probs[0]*100), 2),
                    "Malignant Score": round(float(probs[1]*100), 2),
                    "Selected Class Score": round(score, 2),
                    "File Hash": hashlib.sha256(data).hexdigest()
                }
                st.session_state.history.insert(0, result)

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Classification</div>
                    <div class="result-class">{name}</div>
                    <div class="result-score">{score:.1f}%</div>
                    <div class="result-label">Confidence score</div>
                </div>
                """, unsafe_allow_html=True)

                s1, s2 = st.columns(2)
                with s1:
                    st.metric("Benign", f"{probs[0]*100:.1f}%")
                    st.progress(float(probs[0]))
                with s2:
                    st.metric("Malignant", f"{probs[1]*100:.1f}%")
                    st.progress(float(probs[1]))

                pdf = generate_analysis_pdf(result, image, probs, shape, model)
                st.download_button(
                    "Download PDF Report",
                    data=pdf,
                    file_name=f"BreastScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                disclaimer()

        except UnidentifiedImageError:
            st.error("Invalid image file.")
        except Exception as e:
            st.error("Analysis failed.")
            st.code(str(e))

# ============================================================
# HISTORY
# ============================================================
elif st.session_state.page == "history":
    top_nav()
    st.markdown("### Prediction History")
    st.caption("Results from the current session.")
    st.write("")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        show = df.drop(columns=["File Hash"], errors="ignore")
        st.dataframe(show, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Export CSV",
                data=df.to_csv(index=False).encode(),
                file_name="BreastScan_History.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No analyses yet. Go to Scanner to get started.")

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page == "about":
    top_nav()
    st.markdown("### About the Project")
    st.write("")

    st.markdown("""
    <div class="card">
        <div class="card-title">BreastScan AI</div>
        <div class="card-desc">
            An educational project that demonstrates how a convolutional neural network 
            can classify breast cell images into Benign or Malignant categories.
            Built with TensorFlow, Streamlit and ReportLab.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("**Technology**")
    st.write("Python · TensorFlow · Keras · Streamlit · ReportLab · Pillow · NumPy")

    st.markdown("**Team**")
    st.write("Team Breast Scan AI  \nArmy Public School, LBS Marg, Lucknow  \nAcademic Year 2026–2027")

    st.write("")
    disclaimer()
