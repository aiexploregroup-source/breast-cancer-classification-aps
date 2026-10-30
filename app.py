from pathlib import Path

app_code = r'''import os
import re
import csv
import hashlib
from io import BytesIO, StringIO
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

import tensorflow as tf
import gdown

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as PDFImage, HRFlowable, KeepTogether
)


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BreastScan AI | Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "BreastScan AI"
TEAM_NAME = "Team Breast Scan AI"
SCHOOL_NAME = "Army Public School, LBS Marg, Lucknow"
ACADEMIC_YEAR = "2026–2027"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
GOOGLE_DRIVE_FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"

# IMPORTANT: Confirm these settings against the model's training code.
CLASS_NAMES = ["Benign", "Malignant"]
NORMALIZATION_DIVISOR = 255.0
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

# The app supports a 2-output model (probabilities or logits)
# and a 1-output sigmoid model. See model verification notes below.


# ============================================================
# 2. VISUAL DESIGN: WARM EDITORIAL BACKGROUND
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500;1,600&display=swap" rel="stylesheet">

    <style>
    :root {
        --ink: #26364D;
        --muted: #748398;
        --blue: #4169A8;
        --blue-dark: #2D4F83;
        --teal: #168B78;
        --line: #E5EAF0;
        --paper: #F6F7F5;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Soft layered background with subtle color blooms */
    .stApp {
        background:
            radial-gradient(ellipse at 8% 4%, rgba(195, 216, 244, .48), transparent 31%),
            radial-gradient(ellipse at 96% 16%, rgba(204, 235, 226, .40), transparent 27%),
            radial-gradient(ellipse at 76% 92%, rgba(231, 220, 245, .28), transparent 30%),
            linear-gradient(135deg, #F8F9F7 0%, #F3F6F8 48%, #F8F7F4 100%);
        background-attachment: fixed;
        color: var(--ink);
    }

    /* Very subtle background texture */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .16;
        z-index: 0;
        background-image: radial-gradient(#91A4BC 0.55px, transparent 0.55px);
        background-size: 19px 19px;
        mask-image: linear-gradient(to bottom, black, transparent 85%);
    }

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        font-style: italic !important;
        color: var(--ink);
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: clamp(36px, 5vw, 59px);
        line-height: 1.10;
        letter-spacing: -1.5px;
        font-weight: 500;
        color: #263957;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: #748398;
        line-height: 1.85;
        max-width: 690px;
    }

    .handwritten {
        font-family: 'Caveat', cursive !important;
        font-size: 27px;
        color: #6681A9;
    }

    .eyebrow, .step-label {
        color: #6380A7;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: 30px;
        color: #263957;
        margin: 2px 0 5px 0;
    }

    .section-description {
        font-size: 13px;
        color: #8290A1;
        line-height: 1.75;
    }

    .app-card {
        background: rgba(255,255,255,.88);
        border: 1px solid rgba(224,231,239,.95);
        border-radius: 19px;
        padding: 22px;
        box-shadow: 0 10px 32px rgba(39,61,91,.055);
        backdrop-filter: blur(12px);
        transition: transform .2s ease, box-shadow .2s ease;
        height: 100%;
    }

    .app-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 35px rgba(39,61,91,.085);
    }

    .card-heading {
        font-size: 15px;
        font-weight: 700;
        color: #34445B;
    }

    .card-description {
        font-size: 12px;
        line-height: 1.8;
        color: #8491A1;
        margin-top: 7px;
    }

    .notice {
        background: rgba(237,243,252,.92);
        border: 1px solid #DCE6F3;
        padding: 15px 17px;
        border-radius: 13px;
        color: #506781;
        font-size: 12px;
        line-height: 1.75;
    }

    .safety-notice {
        background: rgba(255,248,237,.96);
        border: 1px solid #F1E3C9;
        padding: 16px;
        border-radius: 13px;
        color: #85663C;
        font-size: 12px;
        line-height: 1.8;
    }

    .result-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #E6EAF0;
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(35,55,85,.045);
    }

    .result-number {
        font-family: 'Playfair Display', serif;
        font-size: 37px;
        color: #355D93;
        font-style: italic;
        font-weight: 500;
        overflow-wrap: anywhere;
    }

    .result-label {
        color: #8793A2;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    .footer {
        text-align: center;
        color: #9AA5B2;
        font-size: 10px;
        padding: 30px 0 12px 0;
        line-height: 1.8;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        min-height: 44px;
        font-weight: 700;
        transition: all .2s ease;
    }

    .stButton > button {
        border: 1px solid var(--blue);
        background: var(--blue);
        color: white;
    }

    .stButton > button:hover {
        background: var(--blue-dark);
        border-color: var(--blue-dark);
        color: white;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        background: var(--teal);
        border: 1px solid var(--teal);
        color: white;
    }

    .stDownloadButton > button:hover {
        background: #107461;
        border-color: #107461;
        color: white;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,.92);
        border-right: 1px solid #E8EDF2;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.72);
        border: 1px dashed #B9C8DA;
        border-radius: 15px;
        padding: 7px;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #E8EDF2;
        padding: 14px;
        border-radius: 14px;
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "Home",
    "history": [],
    "current_analysis": None,
    "show_history_detail": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def safe_text(value):
    """Escape dynamic text before placing it in HTML/ReportLab markup."""
    return escape(str(value))


def safe_filename(filename):
    stem = Path(str(filename)).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
    return stem[:60] or "image"


def image_hash(data):
    return hashlib.sha256(data).hexdigest()


def show_page_heading(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="eyebrow">{safe_text(eyebrow)}</div>
        <div class="section-title">{safe_text(title)}</div>
        <div class="section-description">{safe_text(description)}</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def show_footer():
    st.markdown(
        f"""
        <div class="footer">
            {safe_text(TEAM_NAME)} · {safe_text(ACADEMIC_YEAR)}<br>
            Educational AI project · Not a medical diagnostic tool
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_to(page):
    st.session_state.page = page
    st.rerun()


def open_analysis(record):
    """Open a history record without requiring the original uploader widget."""
    st.session_state.current_analysis = record
    st.session_state.show_history_detail = True
    st.session_state.page = "New analysis"
    st.rerun()


# ============================================================
# 5. IMAGE VALIDATION
# ============================================================

def validate_image_bytes(data):
    if not data:
        return False, "Please choose an image first."

    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File exceeds the {MAX_FILE_SIZE_MB} MB limit."

    try:
        with Image.open(BytesIO(data)) as im:
            if im.format not in ("JPEG", "PNG"):
                return False, "Please upload a JPG, JPEG or PNG image."
            im.verify()
        return True, "Image is valid."
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "The selected file is not a readable image."


def load_pil_image(data):
    with Image.open(BytesIO(data)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


# ============================================================
# 6. MODEL DOWNLOAD AND LOADING
# ============================================================

def is_hdf5_file(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Download model.h5 if needed and load it once per Streamlit process.
    Invalid or incomplete local downloads are removed before retrying.
    """
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1024 or not is_hdf5_file(MODEL_PATH):
        if MODEL_PATH.exists():
            try:
                MODEL_PATH.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot replace invalid model file at {MODEL_PATH}: {exc}"
                )

        try:
            downloaded_path = gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=True,
                fuzzy=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Model download failed. Check the Google Drive file ID, "
                f"file permissions and internet connection. Details: {exc}"
            ) from exc

        if not downloaded_path or not MODEL_PATH.exists():
            raise RuntimeError(
                "Google Drive did not return a model file. "
                "Make sure the file is shared so it can be downloaded."
            )

    if not is_hdf5_file(MODEL_PATH):
        try:
            MODEL_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "The downloaded file is not a valid HDF5 .h5 model. "
            "Your Drive link may point to an HTML preview/error page, "
            "or the model may actually be a .keras file."
        )

    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as exc:
        raise RuntimeError(f"TensorFlow could not load model.h5: {exc}") from exc


# ============================================================
# 7. MODEL INPUT AND PREPROCESSING
# ============================================================

def model_input_details(model):
    shape = model.input_shape

    if isinstance(shape, list):
        raise ValueError("This app currently supports single-input models only.")

    if len(shape) != 4:
        raise ValueError(f"Expected a 4D image input, but model input is {shape}.")

    # Support channels-last (None, H, W, C) and channels-first (None, C, H, W).
    if shape[1] in (1, 3) and shape[-1] not in (1, 3):
        data_format = "channels_first"
        channels, height, width = shape[1], shape[2], shape[3]
    else:
        data_format = "channels_last"
        height, width, channels = shape[1], shape[2], shape[3]

    if height is None or width is None:
        raise ValueError(
            "The model has dynamic image dimensions. This app requires fixed "
            "height and width; set the correct dimensions from training."
        )

    if channels not in (1, 3):
        raise ValueError(f"Unsupported model channel count: {channels}.")

    return int(height), int(width), int(channels), data_format


def preprocess_image(image, model):
    height, width, channels, data_format = model_input_details(model)

    if channels == 1:
        image_for_model = image.convert("L")
    else:
        image_for_model = image.convert("RGB")

    resized = image_for_model.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(resized, dtype=np.float32)

    if channels == 1 and array.ndim == 2:
        array = np.expand_dims(array, axis=-1)

    array = array / float(NORMALIZATION_DIVISOR)

    if data_format == "channels_first":
        array = np.transpose(array, (2, 0, 1))

    array = np.expand_dims(array, axis=0)

    return array, (height, width, channels)


# ============================================================
# 8. OUTPUT INTERPRETATION AND PREDICTION
# ============================================================

def interpret_output(raw_output):
    """
    Supported model outputs:
      * two values: softmax probabilities or two logits
      * one value: sigmoid probability or sigmoid logit

    Assumed order for two outputs: Benign, Malignant.
    Verify this against the model's training code.
    """
    values = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(values)):
        raise ValueError("The model returned NaN or infinite values.")

    if values.size == 2:
        looks_like_probabilities = (
            np.all(values >= 0)
            and np.all(values <= 1)
            and np.isclose(values.sum(), 1.0, atol=0.05)
        )

        if looks_like_probabilities:
            probs = values / values.sum()
        else:
            shifted = values - np.max(values)
            exp_values = np.exp(shifted)
            probs = exp_values / exp_values.sum()

        return probs.astype(float)

    if values.size == 1:
        value = float(values[0])
        if 0.0 <= value <= 1.0:
            malignant = value
        else:
            malignant = 1.0 / (1.0 + np.exp(-np.clip(value, -80, 80)))

        return np.array([1.0 - malignant, malignant], dtype=float)

    raise ValueError(
        f"Expected one or two model outputs; received {values.size}. "
        "This app needs adapting for your model's output shape."
    )


def predict_image(image, model):
    tensor, tensor_shape = preprocess_image(image, model)
    raw = model.predict(tensor, verbose=0)

    # For a single-output model, raw may be shape (1, 1);
    # for a two-output model, commonly shape (1, 2).
    probabilities = interpret_output(raw)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError("Model output count does not match CLASS_NAMES.")

    index = int(np.argmax(probabilities))

    return {
        "result": CLASS_NAMES[index],
        "confidence": float(probabilities[index] * 100.0),
        "probabilities": probabilities.tolist(),
        "tensor_shape": tensor_shape,
        "timestamp": datetime.now(),
    }


# ============================================================
# 9. PDF REPORT GENERATION
# ============================================================

def generate_pdf_report(record, model):
    buffer = BytesIO()
    page_width, _ = A4

    navy = colors.HexColor("#263957")
    blue = colors.HexColor("#4169A8")
    muted = colors.HexColor("#788494")
    line = colors.HexColor("#E3E9F0")
    pale_blue = colors.HexColor("#EDF3FC")
    pale_gray = colors.HexColor("#F5F7FA")
    pale_warning = colors.HexColor("#FFF8ED")

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="BrandSmall",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=blue,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        alignment=TA_LEFT,
        textColor=navy,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=13,
        textColor=muted,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=navy,
        spaceBefore=12,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=13,
        textColor=navy,
    ))
    styles.add(ParagraphStyle(
        name="MutedCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=11,
        textColor=muted,
    ))
    styles.add(ParagraphStyle(
        name="WarningCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=13,
        textColor=colors.HexColor("#805F35"),
    ))

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="BreastScan AI Analysis Report",
        author=TEAM_NAME,
    )

    story = [
        Paragraph("BREASTSCAN AI", styles["BrandSmall"]),
        Paragraph("Image Analysis Report", styles["ReportTitleCustom"]),
        Paragraph(
            "Educational machine-learning project · "
            + safe_text(TEAM_NAME),
            styles["ReportSubCustom"],
        ),
        HRFlowable(width="100%", thickness=.7, color=line, spaceAfter=10),
    ]

    result = safe_text(record["result"])
    confidence = f'{float(record["confidence"]):.2f}%'

    summary = Table(
        [[
            Paragraph(
                f"<b>MODEL OUTPUT</b><br/><font size='17'>{result}</font>",
                styles["BodyCustom"],
            ),
            Paragraph(
                f"<b>MODEL CONFIDENCE</b><br/><font size='17'>{confidence}</font>",
                styles["BodyCustom"],
            ),
        ]],
        colWidths=[84 * mm, 84 * mm],
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale_blue),
        ("BOX", (0, 0), (-1, -1), .7, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 13),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
    ]))
    story.extend([summary, Spacer(1, 5 * mm)])

    story.append(Paragraph("01 · IMAGE INFORMATION", styles["SectionCustom"]))

    image_bytes = record.get("image_bytes")
    if not image_bytes:
        raise ValueError("The original image is not available for this report.")

    image_buffer = BytesIO(image_bytes)
    with Image.open(image_buffer) as source:
        report_image = ImageOps.exif_transpose(source).convert("RGB")
        report_image.thumbnail((650, 650))
        jpeg_buffer = BytesIO()
        report_image.save(jpeg_buffer, format="JPEG", quality=88)
    jpeg_buffer.seek(0)

    # Fit image into a fixed box without stretching.
    preview = PDFImage(jpeg_buffer, width=57 * mm, height=57 * mm)
    preview._restrictSize(57 * mm, 57 * mm)

    timestamp = record["timestamp"]
    if isinstance(timestamp, datetime):
        timestamp_text = timestamp.strftime("%d %B %Y, %I:%M %p")
    else:
        timestamp_text = safe_text(timestamp)

    details = [
        [Paragraph("<b>File name</b>", styles["BodyCustom"]),
         Paragraph(safe_text(record.get("filename", "Uploaded image")), styles["BodyCustom"])],
        [Paragraph("<b>Analysis time</b>", styles["BodyCustom"]),
         Paragraph(safe_text(timestamp_text), styles["BodyCustom"])],
        [Paragraph("<b>Model input</b>", styles["BodyCustom"]),
         Paragraph(" × ".join(map(str, record["tensor_shape"])), styles["BodyCustom"])],
    ]
    details_table = Table(details, colWidths=[30 * mm, 69 * mm])
    details_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), .4, line),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    image_row = Table([[preview, details_table]], colWidths=[64 * mm, 104 * mm])
    image_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(image_row)

    story.append(Paragraph("02 · MODEL OUTPUT BREAKDOWN", styles["SectionCustom"]))

    probability_rows = [[
        Paragraph("<b>Class</b>", styles["BodyCustom"]),
        Paragraph("<b>Model score</b>", styles["BodyCustom"]),
    ]]
    for label, value in zip(CLASS_NAMES, record["probabilities"]):
        probability_rows.append([
            Paragraph(safe_text(label), styles["BodyCustom"]),
            Paragraph(f"{float(value) * 100:.2f}%", styles["BodyCustom"]),
        ])

    probability_table = Table(probability_rows, colWidths=[84 * mm, 84 * mm])
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale_gray),
        ("GRID", (0, 0), (-1, -1), .45, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([
        probability_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "These values are model output scores. They are not validated "
            "medical probabilities, risk estimates or diagnostic measurements.",
            styles["MutedCustom"],
        ),
    ])

    story.append(Paragraph("03 · MODEL INFORMATION", styles["SectionCustom"]))

    model_rows = [
        ["Framework", "TensorFlow / Keras"],
        ["Model file", "model.h5"],
        ["Input shape", str(model.input_shape)],
        ["Configured output labels", ", ".join(CLASS_NAMES)],
        ["Configured normalization", "Pixel values divided by 255"],
    ]
    model_table = Table(model_rows, colWidths=[62 * mm, 106 * mm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), pale_gray),
        ("GRID", (0, 0), (-1, -1), .45, line),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), navy),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(model_table)

    story.append(Paragraph("04 · IMPORTANT LIMITATIONS", styles["SectionCustom"]))

    limitations = [
        "This application is an educational image-classification demonstration.",
        "It is not a clinically validated medical device or diagnostic system.",
        "A model output labelled Benign does not rule out cancer.",
        "A model output labelled Malignant does not confirm cancer.",
        "Image quality, dataset characteristics, training and model limitations can affect outputs.",
        "Do not use this report to make medical decisions or replace professional medical assessment.",
    ]
    for item in limitations:
        story.append(Paragraph("• " + safe_text(item), styles["BodyCustom"]))
        story.append(Spacer(1, 2 * mm))

    warning = Table([[
        Paragraph(
            "<b>EDUCATIONAL USE ONLY</b><br/>"
            "This report is generated by an experimental academic AI project. "
            "It is not a medical diagnosis. For health concerns, consult a "
            "qualified healthcare professional.",
            styles["WarningCustom"],
        )
    ]], colWidths=[168 * mm])
    warning.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale_warning),
        ("BOX", (0, 0), (-1, -1), .7, colors.HexColor("#F1E3C9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([
        Spacer(1, 4 * mm),
        warning,
        Spacer(1, 8 * mm),
        Paragraph(
            safe_text(TEAM_NAME) + " · " + safe_text(SCHOOL_NAME),
            styles["MutedCustom"],
        ),
        Paragraph("Academic session " + safe_text(ACADEMIC_YEAR), styles["MutedCustom"]),
    ])

    def draw_page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(line)
        canvas.setLineWidth(.5)
        canvas.line(18 * mm, 15 * mm, page_width - 18 * mm, 15 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 10 * mm, "BREASTSCAN AI · EDUCATIONAL PROJECT")
        canvas.drawRightString(page_width - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 3px 22px 3px;">
            <div style="display:flex;align-items:center;gap:11px;">
                <div style="background:#EDF3FC;color:#4169A8;border-radius:13px;
                            width:42px;height:42px;display:flex;align-items:center;
                            justify-content:center;font-size:20px;">🔬</div>
                <div>
                    <div style="font-weight:800;color:#263957;font-size:15px;">
                        BreastScan AI
                    </div>
                    <div style="font-size:9px;letter-spacing:1.3px;color:#8793A2;">
                        IMAGE CLASSIFICATION
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)

    page_names = ["Home", "New analysis", "My history", "About project"]
    current_index = page_names.index(st.session_state.page) if st.session_state.page in page_names else 0

    selected = st.radio(
        "Navigation",
        page_names,
        index=current_index,
        label_visibility="collapsed",
        key="main_navigation",
    )

    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

    st.divider()
    st.markdown(
        """
        <div style="background:#F1F5FA;padding:14px;border-radius:13px;">
            <div style="font-size:10px;color:#4169A8;font-weight:800;letter-spacing:1px;">
                PROJECT NOTE
            </div>
            <div style="font-size:11px;color:#6F7E91;line-height:1.75;margin-top:7px;">
                Educational AI demonstration only. Outputs are not medical diagnoses.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption("Team Breast Scan AI")
    st.caption("Academic session 2026–27")


# ============================================================
# 11. HOME PAGE
# ============================================================

if st.session_state.page == "Home":
    st.markdown('<div class="eyebrow">WELCOME TO BREASTSCAN AI</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-title">
            Explore the patterns.<br>
            Understand the technology.
        </div>
        <div class="hero-subtitle">
            An educational image-classification experience demonstrating
            how machine learning can process and classify image data.
        </div>
        <div class="handwritten">A little science, a little curiosity.</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="app-card">
                <div class="eyebrow">YOUR WORKSPACE</div>
                <div style="font-family:'Playfair Display',serif;font-style:italic;
                            font-size:27px;color:#263957;margin-top:10px;">
                    Start a new analysis
                </div>
                <div class="card-description">
                    Upload an image, review it and explore the output
                    generated by the machine-learning model.
                </div>
                <div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:22px;">
                    <span style="background:#EDF3FC;color:#4169A8;padding:7px 11px;
                                 border-radius:8px;font-size:10px;">JPG / JPEG / PNG</span>
                    <span style="background:#EAF6F1;color:#168B78;padding:7px 11px;
                                 border-radius:8px;font-size:10px;">Guided workflow</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start new analysis  →", use_container_width=True):
            go_to("New analysis")

    with right:
        st.markdown(
            """
            <div class="app-card">
                <div class="eyebrow">HOW IT WORKS</div>
                <div style="font-family:'Playfair Display',serif;font-style:italic;
                            font-size:25px;color:#263957;margin-top:10px;">
                    A simple process
                </div>
                <div class="card-description">
                    <b>01</b> &nbsp; Upload an image<br><br>
                    <b>02</b> &nbsp; Review and analyze<br><br>
                    <b>03</b> &nbsp; Explore model output<br><br>
                    <b>04</b> &nbsp; Generate a PDF report
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        """
        <div class="safety-notice">
            <b>Important educational notice</b><br>
            BreastScan AI is an academic image-classification demonstration.
            It is not a clinically validated diagnostic system. Its output
            must never be used to make medical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_footer()


# ============================================================
# 12. NEW ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "New analysis":
    show_page_heading(
        "GUIDED WORKFLOW · IMAGE ANALYSIS",
        "Your analysis workspace",
        "Upload an image, review it and inspect the model output.",
    )

    # Show history selection independently of the uploader.
    if st.session_state.show_history_detail and st.session_state.current_analysis:
        record = st.session_state.current_analysis
        st.info("You are viewing a saved result from this session.")
        try:
            image = load_pil_image(record["image_bytes"])
            st.image(image, caption=record.get("filename", "Previously analyzed image"),
                     use_container_width=True)

            a, b = st.columns(2)
            with a:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">MODEL OUTPUT</div>'
                    f'<div class="result-number">{safe_text(record["result"])}</div></div>',
                    unsafe_allow_html=True,
                )
            with b:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">MODEL CONFIDENCE</div>'
                    f'<div class="result-number">{float(record["confidence"]):.2f}%</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("#### Model output breakdown")
            for label, score in zip(CLASS_NAMES, record["probabilities"]):
                st.write(f"**{label}** · {float(score) * 100:.2f}%")
                st.progress(float(np.clip(score, 0, 1)))

            st.markdown(
                '<div class="safety-notice"><b>Educational use only.</b> '
                'This model output is not a medical diagnosis and must not '
                'be used for medical decisions.</div>',
                unsafe_allow_html=True,
            )

            try:
                model = load_model()
                pdf_data = generate_pdf_report(record, model)
                st.download_button(
                    "Download this analysis as PDF",
                    data=pdf_data,
                    file_name=f"BreastScan_AI_{safe_filename(record.get('filename', 'image'))}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="history_pdf_download",
                )
            except Exception as exc:
                st.warning(f"PDF report could not be prepared: {exc}")

            if st.button("Upload another image", use_container_width=True):
                st.session_state.show_history_detail = False
                st.session_state.current_analysis = None
                st.rerun()

        except Exception as exc:
            st.error(f"Could not reopen this saved analysis: {exc}")

    else:
        st.markdown(
            """
            <div class="app-card">
                <div class="step-label">STEP 01 · SELECT YOUR IMAGE</div>
                <div class="card-heading" style="margin-top:8px;">Upload an image</div>
                <div class="card-description">
                    Choose a JPG, JPEG or PNG image. Maximum file size: 10 MB.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Choose an image",
            type=ALLOWED_EXTENSIONS,
            help="Supported formats: JPG, JPEG and PNG. Maximum file size: 10 MB.",
            key="new_analysis_uploader",
        )

        if uploaded is None:
            st.info("Choose an image to begin. Your selected image will be previewed before analysis.")
        else:
            file_bytes = uploaded.getvalue()
            valid, message = validate_image_bytes(file_bytes)

            if not valid:
                st.error(message)
            else:
                try:
                    image = load_pil_image(file_bytes)

                    st.write("")
                    st.markdown('<div class="step-label">STEP 02 · REVIEW YOUR IMAGE</div>',
                                unsafe_allow_html=True)

                    preview_col, info_col = st.columns([1.15, 1], gap="large")

                    with preview_col:
                        st.image(image, caption="Uploaded image preview", use_container_width=True)

                    with info_col:
                        st.markdown(
                            """
                            <div class="app-card">
                                <div class="card-heading">Image details</div>
                                <div class="card-description">
                                    Check that the selected file is the intended image.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.write(f"**File:** {safe_text(uploaded.name)}")
                        st.write(f"**Dimensions:** {image.width} × {image.height}")
                        st.write(f"**Format:** {safe_text(uploaded.type or 'Image')}")
                        st.write(f"**Size:** {len(file_bytes) / 1024:.1f} KB")

                    st.write("")
                    if st.button("Analyze image  →", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Loading the model and processing your image..."):
                                model = load_model()
                                prediction = predict_image(image, model)

                                record = {
                                    **prediction,
                                    "filename": uploaded.name,
                                    "image_bytes": file_bytes,
                                    "image_hash": image_hash(file_bytes),
                                }

                                st.session_state.current_analysis = record
                                st.session_state.show_history_detail = False

                                # Avoid duplicate history entries on repeated reruns.
                                existing_hashes = {
                                    item.get("image_hash") for item in st.session_state.history
                                }
                                if record["image_hash"] not in existing_hashes:
                                    st.session_state.history.insert(0, record)
                                    st.session_state.history = st.session_state.history[:20]

                            st.success("Analysis completed.")
                            st.rerun()

                        except Exception as exc:
                            st.error("The analysis could not be completed.")
                            st.exception(exc)

                except Exception as exc:
                    st.error(f"Unable to display the selected image: {exc}")

        # Display latest analysis even after Streamlit reruns the page.
        current = st.session_state.current_analysis
        if current and not st.session_state.show_history_detail:
            st.divider()
            st.markdown('<div class="step-label">STEP 03 · YOUR RESULTS</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-title">Analysis results</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="section-description">The following information represents '
                'the output generated by the model.</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            result_col, confidence_col = st.columns(2)
            with result_col:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">MODEL OUTPUT</div>'
                    f'<div class="result-number">{safe_text(current["result"])}</div></div>',
                    unsafe_allow_html=True,
                )
            with confidence_col:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">MODEL CONFIDENCE</div>'
                    f'<div class="result-number">{float(current["confidence"]):.2f}%</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("#### Model output breakdown")
            for label, score in zip(CLASS_NAMES, current["probabilities"]):
                st.write(f"**{label}** · {float(score) * 100:.2f}%")
                st.progress(float(np.clip(score, 0, 1)))

            st.markdown(
                """
                <div class="safety-notice">
                    <b>Please remember</b><br>
                    These are experimental model output scores, not validated
                    medical probabilities. This app cannot establish whether
                    a person has or does not have cancer. Do not use it for
                    medical decisions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown('<div class="step-label">STEP 04 · SAVE YOUR REPORT</div>',
                        unsafe_allow_html=True)

            try:
                model = load_model()
                pdf_bytes = generate_pdf_report(current, model)
                st.download_button(
                    "Download professional PDF report",
                    data=pdf_bytes,
                    file_name=(
                        "BreastScan_AI_Report_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S")
                        + ".pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                    key="current_pdf_download",
                )
            except Exception as exc:
                st.error(f"PDF report generation failed: {exc}")

    show_footer()


# ============================================================
# 13. HISTORY PAGE
# ============================================================

elif st.session_state.page == "My history":
    show_page_heading(
        "YOUR WORKSPACE",
        "Analysis history",
        "Review analyses completed during this Streamlit session.",
    )

    history = st.session_state.history

    if not history:
        st.markdown(
            """
            <div class="app-card" style="text-align:center;padding:42px 20px;">
                <div style="font-size:34px;">📂</div>
                <div style="font-family:'Playfair Display',serif;font-style:italic;
                            font-size:24px;color:#263957;margin-top:10px;">
                    Nothing here yet
                </div>
                <div class="card-description">
                    Completed analyses will appear here during this session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start your first analysis"):
            go_to("New analysis")
    else:
        st.caption(f"{len(history)} recent analysis record(s)")

        for index, item in enumerate(history):
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
                with c1:
                    st.markdown(f"**{safe_text(item.get('filename', 'Image'))}**")
                    st.caption(item["timestamp"].strftime("%d %B %Y · %I:%M %p"))
                with c2:
                    st.markdown(f"**{safe_text(item['result'])}**")
                    st.caption("Model output")
                with c3:
                    st.markdown(f"**{float(item['confidence']):.2f}%**")
                    st.caption("Model confidence")

                if st.button("Open analysis", key=f"open_history_{index}"):
                    open_analysis(item)

        st.write("")
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Filename", "Model output", "Confidence", "Timestamp"])
        for item in history:
            writer.writerow([
                item.get("filename", ""),
                item.get("result", ""),
                f"{float(item.get('confidence', 0)):.2f}",
                item["timestamp"].isoformat() if isinstance(item["timestamp"], datetime)
                else str(item["timestamp"]),
            ])

        st.download_button(
            "Download history as CSV",
            data=csv_buffer.getvalue(),
            file_name="BreastScan_AI_History.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("Clear session history", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_analysis = None
            st.session_state.show_history_detail = False
            st.rerun()

        st.caption(
            "History is temporary and session-based. It is not a permanent "
            "patient record or a secure medical record system."
        )

    show_footer()


# ============================================================
# 14. ABOUT PAGE
# ============================================================

elif st.session_state.page == "About project":
    show_page_heading(
        "ABOUT THE PROJECT",
        "Technology with a purpose.",
        "Learn about the project, its workflow and its limitations.",
    )

    st.markdown(
        f"""
        <div class="app-card">
            <div class="eyebrow">ACADEMIC PROJECT · {safe_text(ACADEMIC_YEAR)}</div>
            <div style="font-family:'Playfair Display',serif;font-style:italic;
                        font-size:30px;color:#263957;margin-top:10px;">
                BreastScan AI
            </div>
            <div class="card-description">
                BreastScan AI is an academic machine-learning project
                demonstrating image preprocessing, model inference and
                an interactive Streamlit application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="app-card">
                <div class="card-heading">Image preprocessing</div>
                <div class="card-description">
                    The uploaded image is converted into the model's expected
                    color format, resized and normalized before inference.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="app-card">
                <div class="card-heading">Machine learning</div>
                <div class="card-description">
                    A trained Keras model processes the image and returns
                    output scores for the configured classes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Technical information")

    details = [
        ("Application", "Streamlit"),
        ("Model framework", "TensorFlow / Keras"),
        ("Model file", "model.h5"),
        ("Image input", "RGB or grayscale according to model shape"),
        ("Configured normalization", "Pixel values divided by 255"),
        ("Output labels", "Benign / Malignant"),
        ("PDF generation", "ReportLab"),
    ]

    for label, value in details:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption(label)
        with col2:
            st.write(value)

    st.write("")
    st.markdown("### Our team")
    st.markdown(
        f"""
        <div class="app-card">
            <div class="eyebrow">TEAM & INSTITUTION</div>
            <div style="font-family:'Playfair Display',serif;font-style:italic;
                        font-size:24px;color:#263957;margin-top:10px;">
                {safe_text(TEAM_NAME)}
            </div>
            <div class="card-description">
                {safe_text(SCHOOL_NAME)}<br>
                Academic session {safe_text(ACADEMIC_YEAR)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <div class="safety-notice">
            <b>Medical and scientific disclaimer</b><br><br>
            This project is intended for educational and demonstration
            purposes only. It has not been established as a clinically
            validated diagnostic tool.<br><br>
            Model predictions may be affected by image quality, dataset
            limitations, training procedures and technical factors.<br><br>
            A Benign output does not rule out cancer, and a Malignant output
            does not confirm cancer. Never use this application to make
            medical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_footer()
'''

out = Path("/mnt/data/app.py")
out.write_text(app_code, encoding="utf-8")

requirements = """streamlit>=1.40.0
tensorflow>=2.15.0
numpy>=1.24.0
Pillow>=10.0.0
gdown>=5.2.0
reportlab>=4.2.0
"""
Path("/mnt/data/requirements.txt").write_text(requirements, encoding="utf-8")

print(f"Created {out} ({out.stat().st_size:,} bytes)")
print(f"Created /mnt/data/requirements.txt")
