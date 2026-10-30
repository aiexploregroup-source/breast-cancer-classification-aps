
import os
import io
import hashlib
import textwrap
from datetime import datetime
from xml.sax.saxutils import escape

import streamlit as st
import tensorflow as tf
import numpy as np
import gdown

from PIL import Image, ImageOps

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage,
    Table, TableStyle, HRFlowable, Flowable
)
from reportlab.lib.utils import ImageReader


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Scan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "model.h5"

IMAGE_SIZE = (48, 48)
CLASS_NAMES = ["Benign", "Malignant"]


# ============================================================
# HTML RENDERING FIX
# ============================================================

def render_html(content):
    """Render HTML safely without accidental Markdown code blocks."""
    st.markdown(
        textwrap.dedent(content).strip(),
        unsafe_allow_html=True
    )


# ============================================================
# CSS
# ============================================================

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #081321;
    --panel: #101F31;
    --teal: #26D5BE;
    --text: #F1F6FC;
    --muted: #91A6BC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(38,213,190,.11), transparent 35%),
        radial-gradient(ellipse at 90% 40%, rgba(42,111,190,.08), transparent 35%),
        var(--bg);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0 22px;
    border-bottom: 1px solid #20334A;
    margin-bottom: 30px;
}

.brand {
    font-size: 20px;
    font-weight: 900;
    letter-spacing: -.7px;
    color: #F4F8FF;
}

.brand span, .hero h1 span {
    color: var(--teal);
}

.brand-sub {
    font-size: 9px;
    color: #829CB4;
    letter-spacing: 2px;
    margin-top: 3px;
}

.online {
    display: inline-block;
    border: 1px solid #176B5D;
    background: #10382F;
    color: #6CE5C5;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 10px;
    font-weight: 800;
}

.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 85% 30%, rgba(38,213,190,.20), transparent 34%),
        linear-gradient(125deg, #142D44, #0C1929 75%);
    border: 1px solid #29475D;
    border-radius: 25px;
    padding: 55px 48px;
    margin-bottom: 28px;
}

.hero:after {
    content: "✳";
    position: absolute;
    right: 9%;
    top: 10%;
    color: rgba(75,220,202,.12);
    font-size: 180px;
    pointer-events: none;
}

.eyebrow {
    display: inline-block;
    background: rgba(38,213,190,.10);
    border: 1px solid rgba(38,213,190,.30);
    color: #63E3D0;
    padding: 8px 14px;
    border-radius: 30px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
}

.hero h1 {
    font-size: clamp(38px, 5.5vw, 66px);
    line-height: 1.09;
    font-weight: 900;
    letter-spacing: -3px;
    color: #F5F9FF;
    margin: 24px 0 15px;
}

.hero p {
    color: #A9BCD0;
    max-width: 580px;
    font-size: 15px;
    line-height: 1.9;
}

.hero-small {
    font-size: 11px;
    color: #829DB5;
    letter-spacing: 1px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #F0F6FF;
}

.section-subtitle {
    color: #8FA7BE;
    font-size: 13px;
    margin-top: 5px;
    margin-bottom: 22px;
}

.feature-card {
    min-height: 175px;
    background: linear-gradient(145deg, #152A40, #101D2F);
    border: 1px solid #263E56;
    border-radius: 17px;
    padding: 22px;
    transition: .25s ease;
}

.feature-card:hover {
    border-color: var(--teal);
    transform: translateY(-4px);
}

.feature-icon {
    font-size: 25px;
    margin-bottom: 16px;
}

.feature-title {
    color: #EDF5FF;
    font-size: 14px;
    font-weight: 800;
}

.feature-description {
    color: #8FA6BC;
    font-size: 11px;
    line-height: 1.7;
    margin-top: 8px;
}

.stepper {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin: 20px 0 30px;
}

.step {
    flex: 1;
    border-bottom: 2px solid #253A50;
    padding-bottom: 12px;
    color: #70889F;
    font-size: 11px;
    font-weight: 700;
}

.step.active {
    border-color: var(--teal);
    color: #62E2D0;
}

.step.done {
    border-color: #278B7D;
    color: #5CCEB9;
}

.panel {
    background: linear-gradient(145deg, #14263A, #0F1C2D);
    border: 1px solid #2A4057;
    border-radius: 19px;
    padding: 25px;
    margin-bottom: 18px;
}

.panel-label {
    font-size: 10px;
    font-weight: 800;
    color: #68DCCB;
    letter-spacing: 1.6px;
}

.panel-title {
    font-size: 23px;
    font-weight: 800;
    color: #F1F6FC;
    margin-top: 8px;
}

.muted {
    color: #91A6BC;
    font-size: 13px;
    line-height: 1.8;
}

[data-testid="stFileUploader"] {
    background: #11243A;
    border: 1px dashed #347B83;
    border-radius: 16px;
    padding: 17px;
}

.stButton > button,
.stDownloadButton > button {
    border: 1px solid #2BD7C0;
    background: linear-gradient(100deg, #15A996, #28D5BD);
    color: #061A1B !important;
    border-radius: 11px;
    font-size: 13px;
    font-weight: 800;
    min-height: 46px;
    transition: .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #4BE3CF;
    border-color: #4BE3CF;
    box-shadow: 0 6px 25px rgba(38,213,190,.18);
    transform: translateY(-2px);
}

.result-box {
    border-radius: 18px;
    padding: 26px;
    margin: 12px 0;
}

.benign {
    background: rgba(21,164,119,.10);
    border: 1px solid #238B70;
}

.malignant {
    background: rgba(215,79,94,.10);
    border: 1px solid #A74E5A;
}

.result-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 800;
    color: #A4B9CB;
}

.result-value {
    font-size: 35px;
    font-weight: 900;
    margin-top: 10px;
}

.disclaimer {
    border: 1px solid #8C6A3C;
    background: rgba(150,105,36,.10);
    color: #E8C995;
    border-radius: 13px;
    padding: 17px;
    font-size: 12px;
    line-height: 1.8;
}

.footer {
    text-align: center;
    border-top: 1px solid #24374C;
    margin-top: 65px;
    padding: 25px 10px;
    color: #70879E;
    font-size: 11px;
    line-height: 2;
}

@media(max-width: 700px) {
    .hero { padding: 32px 23px; }
    .hero h1 { font-size: 40px; letter-spacing: -1.5px; }
    .panel { padding: 18px; }
    .step { font-size: 9px; }
}
</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():
    defaults = {
        "stage": "home",
        "image_bytes": None,
        "image_name": None,
        "image_hash": None,
        "prediction": None,
        "pdf_bytes": None,
        "report_id": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def go_to(stage):
    st.session_state.stage = stage
    st.rerun()


def reset_workflow():
    for key in [
        "image_bytes", "image_name", "image_hash",
        "prediction", "pdf_bytes", "report_id"
    ]:
        st.session_state[key] = None

    st.session_state.stage = "home"
    st.rerun()


# ============================================================
# HEADER
# ============================================================

render_html("""
<div class="topbar">
    <div>
        <div class="brand">🔬 BREAST SCAN <span>AI</span></div>
        <div class="brand-sub">INTELLIGENT IMAGE CLASSIFICATION</div>
    </div>
    <div class="online">● SYSTEM READY</div>
</div>
""")


# ============================================================
# WORKFLOW STEPPER
# ============================================================

STAGES = ["home", "upload", "report", "export"]
stage_index = STAGES.index(st.session_state.stage)

if stage_index > 0:
    labels = [
        "01 INTRODUCTION",
        "02 IMAGE UPLOAD",
        "03 AI REPORT",
        "04 EXPORT"
    ]

    step_html = '<div class="stepper">'

    for i, label in enumerate(labels):
        css = "done" if i < stage_index else (
            "active" if i == stage_index else ""
        )
        step_html += f'<div class="step {css}">{label}</div>'

    step_html += "</div>"
    render_html(step_html)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        temp_path = MODEL_PATH + ".download"

        try:
            downloaded = gdown.download(
                MODEL_URL,
                temp_path,
                quiet=False,
                fuzzy=True
            )

            if not downloaded or not os.path.exists(temp_path):
                raise RuntimeError("Google Drive model download failed.")

            if os.path.getsize(temp_path) == 0:
                raise RuntimeError("Downloaded model file is empty.")

            os.replace(temp_path, MODEL_PATH)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0

        return np.expand_dims(array, axis=0)

    except Exception as exc:
        raise ValueError(f"Image preprocessing failed: {exc}") from exc


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_image(image_bytes):
    model = load_model()
    processed = preprocess_image(image_bytes)

    raw_output = model.predict(processed, verbose=0)
    raw = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    if raw.size not in (1, 2):
        raise ValueError(
            f"Expected 1 or 2 model outputs; received {raw.size}."
        )

    if not np.all(np.isfinite(raw)):
        raise ValueError("Model returned invalid numerical values.")

    if raw.size == 2:
        if (
            np.all(raw >= 0)
            and np.all(raw <= 1)
            and np.isclose(np.sum(raw), 1.0, atol=0.01)
        ):
            probabilities = raw / np.sum(raw)
        else:
            shifted = raw - np.max(raw)
            exp = np.exp(shifted)
            probabilities = exp / np.sum(exp)

    else:
        # Assumption: the single sigmoid output represents Malignant.
        malignant = float(raw[0])

        if not 0 <= malignant <= 1:
            malignant = 1 / (
                1 + np.exp(-np.clip(malignant, -80, 80))
            )

        probabilities = np.array([1 - malignant, malignant])

    index = int(np.argmax(probabilities))

    return {
        "class_index": index,
        "class_name": CLASS_NAMES[index],
        "confidence": float(probabilities[index] * 100),
        "benign_probability": float(probabilities[0] * 100),
        "malignant_probability": float(probabilities[1] * 100)
    }


# ============================================================
# PDF PROBABILITY BARS
# ============================================================

class ProbabilityBars(Flowable):
    def __init__(self, benign, malignant):
        super().__init__()
        self.benign = benign
        self.malignant = malignant
        self.width = 170 * mm
        self.height = 31 * mm

    def draw(self):
        canvas = self.canv

        values = [
            ("Benign", self.benign, colors.HexColor("#159E83")),
            ("Malignant", self.malignant, colors.HexColor("#D75C6A"))
        ]

        y = self.height - 8 * mm

        for label, value, color in values:
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.HexColor("#263B50"))
            canvas.drawString(0, y, label)
            canvas.drawRightString(self.width, y, f"{value:.2f}%")

            y -= 4 * mm

            canvas.setFillColor(colors.HexColor("#E8EDF2"))
            canvas.roundRect(
                0, y, self.width, 3 * mm, 1.5 * mm,
                fill=1, stroke=0
            )

            canvas.setFillColor(color)
            canvas.roundRect(
                0, y,
                self.width * max(0, min(100, value)) / 100,
                3 * mm, 1.5 * mm,
                fill=1, stroke=0
            )

            y -= 9 * mm


# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf(image_bytes, image_name, result):
    buffer = io.BytesIO()
    now = datetime.now()
    report_id = now.strftime("BSA-%Y%m%d-%H%M%S")

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title="Breast Scan AI - Analysis Report",
        author="Team Breast Scan AI"
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        textColor=colors.HexColor("#10283D"),
        alignment=0,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name="SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#137E79"),
        spaceBefore=12,
        spaceAfter=9
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=15,
        textColor=colors.HexColor("#405469")
    ))

    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        textColor=colors.HexColor("#71869A")
    ))

    styles.add(ParagraphStyle(
        name="DisclaimerCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=13,
        textColor=colors.HexColor("#815F29")
    ))

    story = []

    story.append(Paragraph(
        '<font color="#159E92">BREAST SCAN</font> AI',
        styles["Heading2"]
    ))

    story.append(Paragraph(
        "INTELLIGENT IMAGE CLASSIFICATION SYSTEM",
        styles["SmallCustom"]
    ))

    story.append(Spacer(1, 9 * mm))

    story.append(Paragraph(
        "AI Analysis Report",
        styles["ReportTitleCustom"]
    ))

    story.append(Paragraph(
        "Experimental binary image classification report",
        styles["SmallCustom"]
    ))

    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#D9E5E9"),
        spaceAfter=5 * mm
    ))

    metadata = [
        [
            Paragraph("<b>REPORT ID</b>", styles["SmallCustom"]),
            Paragraph("<b>GENERATED ON</b>", styles["SmallCustom"])
        ],
        [
            Paragraph(report_id, styles["BodyCustom"]),
            Paragraph(
                now.strftime("%d %B %Y, %I:%M %p"),
                styles["BodyCustom"]
            )
        ]
    ]

    metadata_table = Table(metadata, colWidths=[82 * mm, 82 * mm])
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F7F8")),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#DCE7EA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(metadata_table)

    story.append(Paragraph(
        "01  CLASSIFICATION RESULT",
        styles["SectionCustom"]
    ))

    is_benign = result["class_index"] == 0
    result_color = "#168C70" if is_benign else "#C24C5C"
    result_background = "#EAF7F2" if is_benign else "#FFF0F1"

    result_table = Table([[
        Paragraph(
            f'<font color="{result_color}"><b>{escape(result["class_name"].upper())}</b></font>',
            ParagraphStyle(
                "ResultPDFCustom",
                fontName="Helvetica-Bold",
                fontSize=21,
                leading=26
            )
        ),
        Paragraph(
            f'<b>{result["confidence"]:.2f}%</b><br/>'
            '<font size="8" color="#71869A">Model confidence score</font>',
            ParagraphStyle(
                "ConfidencePDFCustom",
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=25,
                alignment=2
            )
        )
    ]], colWidths=[82 * mm, 82 * mm])

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(result_background)),
        ("BOX", (0, 0), (-1, -1), .7, colors.HexColor(result_color)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 15),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(result_table)

    story.append(Paragraph(
        "02  CLASS PROBABILITY DISTRIBUTION",
        styles["SectionCustom"]
    ))

    story.append(Paragraph(
        "These values represent the model's output scores for the two classes.",
        styles["BodyCustom"]
    ))

    story.append(Spacer(1, 4 * mm))

    story.append(ProbabilityBars(
        result["benign_probability"],
        result["malignant_probability"]
    ))

    story.append(Paragraph(
        "03  UPLOADED IMAGE",
        styles["SectionCustom"]
    ))

    safe_filename = escape(os.path.basename(image_name))

    with Image.open(io.BytesIO(image_bytes)) as original:
        original = ImageOps.exif_transpose(original)
        original_width, original_height = original.size
        preview = original.convert("RGB")
        preview.thumbnail((1000, 1000))

        image_buffer = io.BytesIO()
        preview.save(image_buffer, format="JPEG", quality=90)
        image_buffer.seek(0)

    story.append(Paragraph(
        f"<b>File name:</b> {safe_filename}",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        f"<b>Original dimensions:</b> {original_width} × {original_height} pixels",
        styles["BodyCustom"]
    ))

    story.append(Paragraph(
        "<b>Model input:</b> 48 × 48 pixels, RGB",
        styles["BodyCustom"]
    ))

    reader = ImageReader(image_buffer)
    image_width, image_height = reader.getSize()

    scale = min(
        (90 * mm) / image_width,
        (65 * mm) / image_height
    )

    pdf_image = PDFImage(
        image_buffer,
        width=image_width * scale,
        height=image_height * scale
    )
    pdf_image.hAlign = "LEFT"
    story.append(Spacer(1, 4 * mm))
    story.append(pdf_image)

    story.append(Paragraph(
        "04  MODEL INFORMATION",
        styles["SectionCustom"]
    ))

    model_info = [
        ["Architecture", "Convolutional Neural Network (CNN)"],
        ["Framework", "TensorFlow / Keras"],
        ["Input format", "RGB image"],
        ["Input resolution", "48 × 48 pixels"],
        ["Output classes", "Benign / Malignant"]
    ]

    model_table = Table(model_info, colWidths=[48 * mm, 116 * mm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F5")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#405469")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#DCE6EA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(model_table)

    story.append(Paragraph(
        "05  IMPORTANT MEDICAL DISCLAIMER",
        styles["SectionCustom"]
    ))

    disclaimer = (
        "<b>Educational use only.</b> This report contains the output of "
        "an experimental machine-learning classification model. It is not "
        "a medical diagnosis, does not establish or exclude cancer, and "
        "must not be used to make healthcare decisions. Model confidence "
        "scores are not clinically validated probabilities. Any medical "
        "concern should be evaluated by a qualified healthcare professional."
    )

    disclaimer_table = Table(
        [[Paragraph(disclaimer, styles["DisclaimerCustom"])]],
        colWidths=[164 * mm]
    )

    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E9")),
        ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#E8CE9A")),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11)
    ]))

    story.append(disclaimer_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(
        "BREAST SCAN AI  |  TEAM BREAST SCAN AI",
        ParagraphStyle(
            "FooterPDFCustom",
            parent=styles["SmallCustom"],
            alignment=1
        )
    ))

    story.append(Paragraph(
        "Army Public School, LBS Marg, Lucknow | 2026",
        ParagraphStyle(
            "FooterPDFCustom2",
            parent=styles["SmallCustom"],
            alignment=1
        )
    ))

    def draw_page(canvas, document):
        canvas.saveState()
        page_width, page_height = A4

        canvas.setFillColor(colors.HexColor("#159E92"))
        canvas.rect(
            0, page_height - 5 * mm,
            page_width, 5 * mm,
            fill=1, stroke=0
        )

        canvas.setStrokeColor(colors.HexColor("#DCE6EA"))
        canvas.line(
            19 * mm, 14 * mm,
            page_width - 19 * mm, 14 * mm
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#8497A9"))
        canvas.drawString(
            19 * mm, 9 * mm,
            "BREAST SCAN AI | EDUCATIONAL REPORT"
        )
        canvas.drawRightString(
            page_width - 19 * mm, 9 * mm,
            f"PAGE {document.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)

    buffer.seek(0)
    return buffer.getvalue(), report_id


# ============================================================
# LANDING PAGE
# ============================================================

if st.session_state.stage == "home":

    render_html("""
    <div class="hero">
        <div class="eyebrow">✦ DEEP LEARNING • IMAGE CLASSIFICATION</div>

        <h1>
            Understand Images.<br>
            <span>Powered by AI.</span>
        </h1>

        <p>
            Explore an experimental breast cell image classification
            system powered by a custom Convolutional Neural Network.
            Upload an image, run the model and generate a structured
            analysis report.
        </p>

        <div class="hero-small">
            TENSORFLOW • CNN • PYTHON • STREAMLIT
        </div>
    </div>
    """)

    _, cta, _ = st.columns([1, 2, 1])

    with cta:
        if st.button("START IMAGE ANALYSIS  →", use_container_width=True):
            go_to("upload")

    st.write("")

    render_html("""
    <div class="section-title">A simple four-step experience.</div>
    <div class="section-subtitle">
        Everything you need, from image upload to a downloadable report.
    </div>
    """)

    features = [
        ("📤", "Upload", "Select a JPG, JPEG or PNG image."),
        ("🖼️", "Preview", "Inspect your image before analysis."),
        ("🧠", "AI Analysis", "Run the trained CNN classification model."),
        ("📄", "PDF Report", "Generate and download a professional report.")
    ]

    columns = st.columns(4)

    for col, (icon, title, description) in zip(columns, features):
        with col:
            render_html(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">{description}</div>
            </div>
            """)

    st.write("")

    render_html("""
    <div class="panel">
        <div class="panel-label">THE TECHNOLOGY</div>
        <div class="panel-title">Behind the intelligence.</div>
        <p class="muted">
            A custom Convolutional Neural Network processes image data
            after RGB conversion, resizing and pixel normalization.
            The model then generates a binary classification output.
        </p>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:20px;">
            <span class="online">TENSORFLOW / KERAS</span>
            <span class="online">48 × 48 RGB</span>
            <span class="online">2 CLASSES</span>
        </div>
    </div>

    <div class="disclaimer">
        <b>Educational project notice:</b>
        This experimental model is intended for demonstrating image
        classification, not for medical diagnosis.
    </div>
    """)


# ============================================================
# IMAGE UPLOAD
# ============================================================

elif st.session_state.stage == "upload":

    render_html("""
    <div class="panel">
        <div class="panel-label">STEP 02 / IMAGE INPUT</div>
        <div class="panel-title">Upload your image.</div>
        <p class="muted">
            Choose an image to prepare it for the classification workflow.
        </p>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Drag and drop an image here",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
        key="image_uploader"
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        if image_hash != st.session_state.image_hash:
            st.session_state.image_bytes = image_bytes
            st.session_state.image_name = uploaded_file.name
            st.session_state.image_hash = image_hash
            st.session_state.prediction = None
            st.session_state.pdf_bytes = None

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image).convert("RGB")

            render_html("### Image Preview")

            image_col, info_col = st.columns([1.25, .75])

            with image_col:
                st.image(
                    image,
                    caption="Your uploaded image",
                    use_container_width=True
                )

            with info_col:
                render_html("""
                <div class="panel">
                    <div class="panel-label">IMAGE DETAILS</div>
                """)

                st.write("**Filename**")
                st.write(uploaded_file.name)

                st.write("**Dimensions**")
                st.write(f"{image.width} × {image.height} pixels")

                st.write("**Format**")
                st.write(uploaded_file.type or "Image")

                st.write("**Model input**")
                st.write("48 × 48 RGB")

                render_html("</div>")

            render_html("""
            <div class="panel">
                <div class="panel-label">PREPROCESSING</div>
                <div class="panel-title">Ready for analysis</div>
                <p class="muted">
                    The image will be converted to RGB, resized to
                    48 × 48 pixels and normalized before entering
                    the CNN model.
                </p>
            </div>
            """)

            action1, action2 = st.columns(2)

            with action1:
                if st.button("← BACK TO HOME", use_container_width=True):
                    go_to("home")

            with action2:
                if st.button("RUN AI ANALYSIS  →", use_container_width=True):
                    try:
                        with st.spinner("Loading AI model and analyzing image..."):
                            result = predict_image(image_bytes)

                        st.session_state.prediction = result
                        st.session_state.pdf_bytes = None
                        go_to("report")

                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

        except Exception as exc:
            st.error(f"Unable to read image: {exc}")

    else:
        st.info("Upload an image to activate the preview and analysis steps.")

        if st.button("← BACK TO HOME"):
            go_to("home")


# ============================================================
# ANALYSIS REPORT
# ============================================================

elif st.session_state.stage == "report":

    result = st.session_state.prediction

    if result is None or st.session_state.image_bytes is None:
        go_to("upload")

    render_html("""
    <div class="panel">
        <div class="panel-label">STEP 03 / CLASSIFICATION</div>
        <div class="panel-title">Analysis completed.</div>
        <p class="muted">
            The CNN has processed the uploaded image.
            Review the model output below.
        </p>
    </div>
    """)

    is_benign = result["class_index"] == 0
    result_class = "benign" if is_benign else "malignant"
    result_color = "#49D4A7" if is_benign else "#FF8994"

    render_html(f"""
    <div class="result-box {result_class}">
        <div class="result-label">MODEL CLASSIFICATION</div>
        <div class="result-value" style="color:{result_color};">
            ● {escape(result["class_name"])}
        </div>
        <div style="margin-top:12px;color:#A8B9CA;font-size:13px;">
            Model confidence score:
            <b style="color:{result_color};">
                {result["confidence"]:.2f}%
            </b>
        </div>
    </div>
    """)

    render_html("### Probability Breakdown")

    p1, p2 = st.columns(2)

    with p1:
        render_html(f"""
        <div class="panel">
            <div class="panel-label">BENIGN</div>
            <div class="panel-title" style="color:#49D4A7;">
                {result["benign_probability"]:.2f}%
            </div>
        </div>
        """)
        st.progress(result["benign_probability"] / 100)

    with p2:
        render_html(f"""
        <div class="panel">
            <div class="panel-label">MALIGNANT</div>
            <div class="panel-title" style="color:#FF8994;">
                {result["malignant_probability"]:.2f}%
            </div>
        </div>
        """)
        st.progress(result["malignant_probability"] / 100)

    render_html("### Analyzed Image")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            st.session_state.image_bytes,
            caption=st.session_state.image_name,
            use_container_width=True
        )

    with col2:
        render_html("""
        <div class="panel">
            <div class="panel-label">MODEL INFORMATION</div>
            <div class="panel-title">CNN Engine</div>
            <p class="muted">
                Framework: TensorFlow / Keras<br>
                Architecture: CNN<br>
                Input: 48 × 48 RGB<br>
                Classes: 2<br>
                Preprocessing: RGB + Normalization
            </p>
        </div>
        """)

    render_html("""
    <div class="disclaimer">
        <b>Important:</b> This result is an experimental model
        classification, not a medical diagnosis. A model confidence
        score does not establish or exclude cancer. Consult a
        qualified healthcare professional for medical assessment.
    </div>
    """)

    st.write("")

    back_col, next_col = st.columns(2)

    with back_col:
        if st.button("← BACK TO IMAGE", use_container_width=True):
            go_to("upload")

    with next_col:
        if st.button("GENERATE PDF REPORT  →", use_container_width=True):
            try:
                with st.spinner("Designing your PDF report..."):
                    pdf, report_id = generate_pdf(
                        st.session_state.image_bytes,
                        st.session_state.image_name,
                        result
                    )

                st.session_state.pdf_bytes = pdf
                st.session_state.report_id = report_id
                go_to("export")

            except Exception as exc:
                st.error(f"Could not generate PDF: {exc}")


# ============================================================
# PDF EXPORT
# ============================================================

elif st.session_state.stage == "export":

    if st.session_state.pdf_bytes is None:
        go_to("report")

    render_html("""
    <div class="panel">
        <div class="panel-label">STEP 04 / DOCUMENT EXPORT</div>
        <div class="panel-title">Your report is ready.</div>
        <p class="muted">
            Your AI analysis has been converted into a professionally
            formatted PDF document.
        </p>
    </div>

    <div class="panel">
        <div style="display:flex;align-items:center;gap:17px;">
            <div style="font-size:48px;">📄</div>
            <div>
                <div style="font-size:18px;font-weight:800;color:#F0F6FC;">
                    Breast Scan AI Report
                </div>
                <div style="color:#8FA6BC;font-size:12px;margin-top:5px;">
                    PDF DOCUMENT · A4 · PRINT READY
                </div>
            </div>
        </div>
    </div>
    """)

    render_html("### Report contents")

    report_items = [
        "Report identification and generation timestamp",
        "AI classification result",
        "Model confidence score",
        "Benign and Malignant probability bars",
        "Uploaded image and image information",
        "CNN model technical details",
        "Educational medical disclaimer"
    ]

    for item in report_items:
        st.markdown(f"✓ {item}")

    filename = (
        "Breast_Scan_AI_Report_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pdf"
    )

    st.download_button(
        label="⬇ DOWNLOAD PDF REPORT",
        data=st.session_state.pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True
    )

    render_html("""
    <div class="panel">
        <div class="panel-label">PRINT INSTRUCTIONS</div>
        <div class="panel-title">Ready for printing.</div>
        <p class="muted">
            Download the PDF above, open it in your browser or PDF reader,
            and use Ctrl + P to print your report.
        </p>
    </div>
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("← VIEW ANALYSIS", use_container_width=True):
            go_to("report")

    with col2:
        if st.button("＋ NEW ANALYSIS", use_container_width=True):
            reset_workflow()


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">
    <div style="color:#EAF3FC;font-size:15px;font-weight:900;letter-spacing:1px;">
        BREAST SCAN <span style="color:#26D5BE;">AI</span>
    </div>
    <div style="margin-top:8px;">Developed by Team Breast Scan AI</div>
    <div>Army Public School, LBS Marg, Lucknow</div>
    <div style="margin-top:12px;color:#607991;">
        © 2026 · Educational AI Project
    </div>
</div>
""")
