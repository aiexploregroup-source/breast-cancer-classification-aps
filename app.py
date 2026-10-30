
import os
import re
import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

import tensorflow as tf
import gdown

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
    KeepTogether,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BreastScan AI | Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "BreastScan AI"
TEAM_NAME = "Team Breast Scan AI"
ACADEMIC_YEAR = "2026–2027"

MODEL_PATH = Path("model.h5")

GOOGLE_DRIVE_FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"

CLASS_NAMES = ["Benign", "Malignant"]

# This assumes your model was trained using RGB images
# normalized by dividing pixel values by 255.
NORMALIZATION_DIVISOR = 255.0

MAX_FILE_SIZE_MB = 10

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

# Automatic interpretation:
# - Two outputs: softmax if needed.
# - One output: sigmoid probability for Malignant.
MODEL_OUTPUT_MODE = "auto"


# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400;1,500;1,600&display=swap"
    rel="stylesheet">

    <style>

    :root {
        --background: #F7F8F6;
        --surface: #FFFFFF;
        --primary: #4169A8;
        --primary-dark: #294D83;
        --teal: #168B78;
        --text: #26364D;
        --muted: #788494;
        --border: #E5EAF0;
        --soft-blue: #EDF3FC;
        --soft-green: #EAF6F1;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        font-style: italic !important;
        color: var(--text);
    }

    p, label, span, div {
        font-family: 'Inter', sans-serif;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: clamp(36px, 5vw, 58px);
        font-weight: 500;
        line-height: 1.12;
        color: #263957;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #778494;
        line-height: 1.8;
        max-width: 650px;
    }

    .handwritten {
        font-family: 'Caveat', cursive !important;
        font-size: 26px;
        color: #6681A9;
    }

    .eyebrow {
        color: #6380A7;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: 29px;
        color: #263957;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 13px;
        color: #8390A0;
        line-height: 1.7;
    }

    .app-card {
        background: #FFFFFF;
        border: 1px solid #E8EDF2;
        border-radius: 18px;
        padding: 23px;
        box-shadow: 0 5px 20px rgba(35, 55, 85, 0.035);
        transition: all 0.2s ease;
        height: 100%;
    }

    .app-card:hover {
        border-color: #C9D8EC;
        box-shadow: 0 10px 28px rgba(35, 55, 85, 0.07);
        transform: translateY(-2px);
    }

    .card-heading {
        font-size: 15px;
        font-weight: 700;
        color: #34445B;
    }

    .card-description {
        font-size: 12px;
        line-height: 1.7;
        color: #8793A2;
        margin-top: 7px;
    }

    .step-label {
        color: #6380A7;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
    }

    .notice {
        background: #EEF3FA;
        border: 1px solid #DCE6F3;
        padding: 15px 17px;
        border-radius: 13px;
        color: #506781;
        font-size: 12px;
        line-height: 1.7;
    }

    .safety-notice {
        background: #FFF8ED;
        border: 1px solid #F1E3C9;
        padding: 16px;
        border-radius: 13px;
        color: #85663C;
        font-size: 12px;
        line-height: 1.8;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #E6EAF0;
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(35, 55, 85, 0.035);
    }

    .result-number {
        font-family: 'Playfair Display', serif;
        font-size: 40px;
        color: #355D93;
        font-style: italic;
        font-weight: 500;
    }

    .result-label {
        color: #8793A2;
        font-size: 12px;
        font-weight: 600;
    }

    .progress-track {
        background: #EDF0F4;
        height: 9px;
        border-radius: 20px;
        overflow: hidden;
    }

    .footer {
        text-align: center;
        color: #A0A9B4;
        font-size: 11px;
        padding: 30px 0 12px 0;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid #4169A8;
        background: #4169A8;
        color: white;
        font-weight: 700;
        padding: 11px 19px;
        min-height: 44px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #294D83;
        border-color: #294D83;
        color: white;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        border-radius: 11px;
        background: #168B78;
        color: white;
        border: none;
        font-weight: 700;
        min-height: 44px;
    }

    .stDownloadButton > button:hover {
        background: #107461;
        color: white;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E8EDF2;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-size: 13px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8EDF2;
        padding: 15px;
        border-radius: 14px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_filename(filename):
    """Make a safe filename for downloads."""
    name = Path(filename).stem
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name[:60] or "image"


def image_to_bytes(image, image_format="PNG"):
    """Convert a PIL image into bytes."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def get_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def validate_uploaded_image(uploaded_file):
    """Validate file size and image readability."""

    if uploaded_file is None:
        return False, "Please select an image first."

    size_mb = uploaded_file.size / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB."

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)

        if image.format not in ["JPEG", "PNG"]:
            return False, "Please upload a JPG, JPEG or PNG image."

        return True, "Image looks valid."

    except (UnidentifiedImageError, OSError, ValueError):
        uploaded_file.seek(0)
        return False, "This file could not be read as an image."


# ============================================================
# 5. MODEL DOWNLOAD AND LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Download and load the Keras model.

    The Google Drive ID must point to the actual model.h5 file.
    """

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1000:

        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        try:
            downloaded = gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
            )

            if not downloaded:
                raise RuntimeError("Google Drive download did not complete.")

        except Exception as error:
            raise RuntimeError(
                f"Could not download the model: {error}"
            )

    # Check for an HDF5 file rather than an HTML error page.
    with open(MODEL_PATH, "rb") as file:
        signature = file.read(8)

    if signature != b"\x89HDF\r\n\x1a\n":
        raise RuntimeError(
            "The downloaded file is not a valid HDF5 .h5 model. "
            "Check your Google Drive file ID and sharing permissions."
        )

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"TensorFlow could not load the model: {error}"
        )

    return model


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def get_model_input_details(model):
    """Read the expected input dimensions from the model."""

    shape = model.input_shape

    if isinstance(shape, list):
        raise ValueError(
            "This app currently supports single-input models only."
        )

    if len(shape) != 4:
        raise ValueError(
            f"Unexpected model input shape: {shape}"
        )

    if shape[1] in (1, 3) and shape[-1] not in (1, 3):
        data_format = "channels_first"
        channels = shape[1]
        height = shape[2]
        width = shape[3]

    else:
        data_format = "channels_last"
        height = shape[1]
        width = shape[2]
        channels = shape[3]

    if height is None or width is None:
        raise ValueError(
            "The model must have fixed image dimensions."
        )

    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported number of channels: {channels}"
        )

    return int(height), int(width), int(channels), data_format


def preprocess_image(image, model):
    """
    Resize and normalize an image.

    IMPORTANT:
    This preprocessing assumes training used RGB images
    with pixel values divided by 255.
    """

    height, width, channels, data_format = get_model_input_details(model)

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(image, dtype=np.float32)

    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)

    array = array / NORMALIZATION_DIVISOR

    if data_format == "channels_first":
        array = np.transpose(array, (2, 0, 1))

    array = np.expand_dims(array, axis=0)

    return array, image, (height, width, channels)


# ============================================================
# 7. PREDICTION
# ============================================================

def interpret_model_output(raw_output):
    """
    Interpret model output.

    Supported:
    - Two-class probability output
    - Two-class logits
    - One-output sigmoid probability
    - One-output sigmoid logit

    Class order is assumed to be Benign, Malignant.
    Verify this against the original training code.
    """

    output = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    if len(output) == 2:

        if not np.all(np.isfinite(output)):
            raise ValueError("The model returned invalid numerical values.")

        if (
            np.all(output >= 0)
            and np.all(output <= 1)
            and np.isclose(np.sum(output), 1.0, atol=0.05)
        ):
            probabilities = output

        else:
            # Treat the two outputs as logits.
            shifted = output - np.max(output)
            exponentials = np.exp(shifted)
            probabilities = exponentials / np.sum(exponentials)

        return probabilities

    elif len(output) == 1:

        value = float(output[0])

        if not np.isfinite(value):
            raise ValueError("The model returned an invalid value.")

        if 0 <= value <= 1:
            malignant_probability = value

        else:
            malignant_probability = 1 / (1 + np.exp(-np.clip(value, -80, 80)))

        return np.array([
            1 - malignant_probability,
            malignant_probability,
        ])

    else:
        raise ValueError(
            f"Expected one or two model outputs, but received {len(output)}."
        )


def run_prediction(image, model):

    processed, preview, tensor_shape = preprocess_image(image, model)

    raw_output = model.predict(
        processed,
        verbose=0,
    )

    probabilities = interpret_model_output(raw_output)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index]) * 100

    return {
        "result": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "tensor_shape": tensor_shape,
        "preview": preview,
        "timestamp": datetime.now(),
    }


# ============================================================
# 8. PROFESSIONAL PDF REPORT
# ============================================================

def generate_analysis_pdf(
    result,
    original_image,
    filename,
    model,
):

    buffer = BytesIO()

    page_width, page_height = A4

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="BreastScan AI Analysis Report",
        author=TEAM_NAME,
    )

    navy = colors.HexColor("#263957")
    blue = colors.HexColor("#4169A8")
    teal = colors.HexColor("#168B78")
    light_blue = colors.HexColor("#EDF3FC")
    light_gray = colors.HexColor("#F5F7FA")
    muted = colors.HexColor("#788494")
    border = colors.HexColor("#E3E9F0")
    warning_bg = colors.HexColor("#FFF8ED")

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=29,
            textColor=navy,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            textColor=muted,
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            textColor=navy,
            spaceBefore=13,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            textColor=navy,
        )
    )

    styles.add(
        ParagraphStyle(
            name="MutedSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=12,
            textColor=muted,
        )
    )

    styles.add(
        ParagraphStyle(
            name="WarningText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=14,
            textColor=colors.HexColor("#805F35"),
        )
    )

    story = []

    # Header
    story.append(
        Paragraph(
            "BREASTSCAN AI",
            ParagraphStyle(
                "Brand",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=10,
                textColor=blue,
                tracking=1.5,
            ),
        )
    )

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            "Image Analysis Report",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            "Educational machine-learning project · Team Breast Scan AI",
            styles["ReportSubtitle"],
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=border,
            spaceAfter=12,
        )
    )

    # Result banner
    prediction_text = escape(str(result["result"]))
    confidence_text = f'{result["confidence"]:.2f}%'

    result_data = [[
        Paragraph(
            f"<b>MODEL OUTPUT</b><br/><font size='18'>{prediction_text}</font>",
            styles["BodySmall"],
        ),
        Paragraph(
            f"<b>MODEL CONFIDENCE</b><br/><font size='18'>{confidence_text}</font>",
            styles["BodySmall"],
        ),
    ]]

    result_table = Table(
        result_data,
        colWidths=[85 * mm, 85 * mm],
    )

    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_blue),
            ("BOX", (0, 0), (-1, -1), 0.7, border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 13),
            ("RIGHTPADDING", (0, 0), (-1, -1), 13),
            ("TOPPADDING", (0, 0), (-1, -1), 15),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
        ])
    )

    story.append(result_table)
    story.append(Spacer(1, 7 * mm))

    # Image preview
    story.append(
        Paragraph(
            "01 · IMAGE INFORMATION",
            styles["SectionHeading"],
        )
    )

    image_buffer = BytesIO()
    original_image.convert("RGB").save(
        image_buffer,
        format="JPEG",
        quality=88,
    )
    image_buffer.seek(0)

    image_preview = PDFImage(
        image_buffer,
        width=65 * mm,
        height=65 * mm,
        kind="proportional",
    )

    filename_text = escape(str(filename))
    timestamp_text = result["timestamp"].strftime("%d %B %Y, %I:%M %p")

    image_info = [
        [
            Paragraph("<b>File name</b>", styles["BodySmall"]),
            Paragraph(filename_text, styles["BodySmall"]),
        ],
        [
            Paragraph("<b>Analysis time</b>", styles["BodySmall"]),
            Paragraph(timestamp_text, styles["BodySmall"]),
        ],
        [
            Paragraph("<b>Model input</b>", styles["BodySmall"]),
            Paragraph(
                " × ".join(map(str, result["tensor_shape"])),
                styles["BodySmall"],
            ),
        ],
        [
            Paragraph("<b>Image format</b>", styles["BodySmall"]),
            Paragraph("RGB / JPEG preview", styles["BodySmall"]),
        ],
    ]

    info_table = Table(
        image_info,
        colWidths=[32 * mm, 68 * mm],
    )

    info_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    preview_table = Table(
        [[image_preview, info_table]],
        colWidths=[70 * mm, 100 * mm],
    )

    preview_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    story.append(preview_table)

    # Probability breakdown
    story.append(
        Paragraph(
            "02 · MODEL OUTPUT BREAKDOWN",
            styles["SectionHeading"],
        )
    )

    probability_data = [[
        Paragraph("<b>Class</b>", styles["BodySmall"]),
        Paragraph("<b>Model score</b>", styles["BodySmall"]),
    ]]

    for name, probability in zip(
        CLASS_NAMES,
        result["probabilities"],
    ):
        probability_data.append([
            Paragraph(escape(name), styles["BodySmall"]),
            Paragraph(
                f"{float(probability) * 100:.2f}%",
                styles["BodySmall"],
            ),
        ])

    probability_table = Table(
        probability_data,
        colWidths=[85 * mm, 85 * mm],
    )

    probability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), light_gray),
            ("GRID", (0, 0), (-1, -1), 0.5, border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ])
    )

    story.append(probability_table)

    story.append(
        Paragraph(
            "The scores represent the model's output for the submitted "
            "image. They are not validated medical probabilities or "
            "measurements of disease risk.",
            styles["MutedSmall"],
        )
    )

    # Page two
    story.append(Spacer(1, 7 * mm))

    story.append(
        Paragraph(
            "03 · MODEL INFORMATION",
            styles["SectionHeading"],
        )
    )

    model_shape = str(model.input_shape)

    model_details = [
        ["Model framework", "TensorFlow / Keras"],
        ["Model file", "model.h5"],
        ["Model input shape", model_shape],
        ["Number of output classes", "2"],
        ["Output labels", "Benign, Malignant"],
        ["Image normalization", "Pixel values divided by 255"],
    ]

    details_table = Table(
        model_details,
        colWidths=[65 * mm, 105 * mm],
    )

    details_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), light_gray),
            ("GRID", (0, 0), (-1, -1), 0.5, border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), navy),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ])
    )

    story.append(details_table)

    # Limitations
    story.append(
        Paragraph(
            "04 · IMPORTANT LIMITATIONS",
            styles["SectionHeading"],
        )
    )

    limitations = [
        "This application is an educational image-classification "
        "demonstration created for an academic project.",

        "It is not a clinically validated medical device or diagnostic "
        "system.",

        "A model output labelled Benign does not rule out cancer. "
        "A model output labelled Malignant does not confirm cancer.",

        "Image quality, dataset characteristics, training methods and "
        "model limitations can affect the output.",

        "This report must not be used to make medical decisions or "
        "replace professional medical assessment.",
    ]

    for item in limitations:
        story.append(
            Paragraph(
                "• " + escape(item),
                styles["BodySmall"],
            )
        )
        story.append(Spacer(1, 3 * mm))

    warning_table = Table(
        [[
            Paragraph(
                "<b>EDUCATIONAL USE ONLY</b><br/>"
                "This report is generated by an experimental academic "
                "AI project. It is not a medical diagnosis. "
                "For health concerns, consult a qualified healthcare professional.",
                styles["WarningText"],
            )
        ]],
        colWidths=[170 * mm],
    )

    warning_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), warning_bg),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#F1E3C9")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ])
    )

    story.append(Spacer(1, 4 * mm))
    story.append(warning_table)

    story.append(Spacer(1, 12 * mm))

    story.append(
        Paragraph(
            f"{escape(TEAM_NAME)} · Army Public School, LBS Marg, Lucknow",
            styles["MutedSmall"],
        )
    )

    story.append(
        Paragraph(
            f"Academic session {escape(ACADEMIC_YEAR)}",
            styles["MutedSmall"],
        )
    )

    def page_decor(canvas, document):
        canvas.saveState()

        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.5)

        canvas.line(
            19 * mm,
            15 * mm,
            page_width - 19 * mm,
            15 * mm,
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)

        canvas.drawString(
            19 * mm,
            10 * mm,
            "BREASTSCAN AI · EDUCATIONAL PROJECT",
        )

        canvas.drawRightString(
            page_width - 19 * mm,
            10 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=page_decor,
        onLaterPages=page_decor,
    )

    return buffer.getvalue()


# ============================================================
# 9. REUSABLE UI COMPONENTS
# ============================================================

def show_page_heading(eyebrow, title, description):

    st.markdown(
        f"""
        <div class="eyebrow">{escape(eyebrow)}</div>
        <div class="section-title">{escape(title)}</div>
        <div class="section-description">{escape(description)}</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")


def show_footer():

    st.markdown(
        f"""
        <div class="footer">
            {escape(TEAM_NAME)} · {escape(ACADEMIC_YEAR)}
            <br>
            Educational AI project · Not a medical diagnostic tool
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate_to(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# 10. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 12px 4px 22px 4px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="
                    background:#EDF3FC;
                    color:#4169A8;
                    border-radius:13px;
                    width:43px;
                    height:43px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:21px;">
                    🔬
                </div>
                <div>
                    <div style="
                        font-weight:800;
                        color:#263957;
                        font-size:16px;">
                        BreastScan AI
                    </div>
                    <div style="
                        font-size:9px;
                        letter-spacing:1.4px;
                        color:#8793A2;">
                        IMAGE CLASSIFICATION
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eyebrow">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    pages = [
        "Home",
        "New analysis",
        "My history",
        "About project",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page),
        label_visibility="collapsed",
    )

    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    st.divider()

    st.markdown(
        """
        <div style="
            background:#F2F6FB;
            padding:15px;
            border-radius:13px;">
            <div style="
                color:#4169A8;
                font-size:11px;
                font-weight:800;
                letter-spacing:1px;">
                PROJECT NOTE
            </div>
            <div style="
                font-size:12px;
                color:#6F7E91;
                line-height:1.7;
                margin-top:7px;">
                This is an educational AI demonstration.
                Its outputs are not medical diagnoses.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption("Made by Team Breast Scan AI")
    st.caption("Academic session 2026–27")


# ============================================================
# 11. HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="eyebrow">WELCOME TO BREASTSCAN AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-title">
            Explore the patterns.<br>
            Understand the technology.
        </div>

        <div class="hero-subtitle">
            An educational image-classification experience
            demonstrating how machine learning can process
            and classify image data.
        </div>

        <div class="handwritten">
            A little science, a little curiosity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:

        st.markdown(
            """
            <div class="app-card">
                <div class="eyebrow">YOUR WORKSPACE</div>

                <div style="
                    font-family:'Playfair Display',serif;
                    font-style:italic;
                    font-size:26px;
                    color:#263957;
                    margin-top:10px;">
                    Start a new analysis
                </div>

                <div class="card-description">
                    Upload an image, review it and explore
                    the output generated by the machine-learning model.
                </div>

                <div style="
                    display:flex;
                    gap:10px;
                    margin-top:24px;
                    flex-wrap:wrap;">
                    <span style="
                        background:#EDF3FC;
                        color:#4169A8;
                        padding:7px 12px;
                        border-radius:8px;
                        font-size:11px;">
                        JPG / JPEG / PNG
                    </span>

                    <span style="
                        background:#EAF6F1;
                        color:#168B78;
                        padding:7px 12px;
                        border-radius:8px;
                        font-size:11px;">
                        Guided workflow
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Start new analysis  →",
            use_container_width=True,
        ):
            navigate_to("New analysis")

    with col2:

        st.markdown(
            """
            <div class="app-card">
                <div class="eyebrow">HOW IT WORKS</div>

                <div style="
                    font-family:'Playfair Display',serif;
                    font-style:italic;
                    font-size:23px;
                    color:#263957;
                    margin-top:10px;">
                    A simple process
                </div>

                <div class="card-description">
                    <b>01</b> &nbsp; Upload an image
                    <br><br>
                    <b>02</b> &nbsp; Review and analyze
                    <br><br>
                    <b>03</b> &nbsp; Explore model output
                    <br><br>
                    <b>04</b> &nbsp; Generate a PDF report
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    st.markdown(
        """
        <div class="safety-notice">
            <b>Important educational notice</b><br>
            BreastScan AI is an academic image-classification
            demonstration. It is not a clinically validated diagnostic
            system. Its output must not be used to make medical decisions.
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
        "GUIDED WORKFLOW · 01",
        "New image analysis",
        "Follow the steps below to explore the model's output.",
    )

    # Workflow indicator
    st.markdown(
        """
        <div style="
            display:flex;
            gap:8px;
            align-items:center;
            margin:15px 0 25px 0;
            flex-wrap:wrap;">

            <span style="
                background:#4169A8;
                color:white;
                padding:7px 13px;
                border-radius:20px;
                font-size:11px;">
                1. Upload
            </span>

            <span style="color:#AAB3BF;">→</span>

            <span style="
                background:#EDF1F5;
                color:#68788B;
                padding:7px 13px;
                border-radius:20px;
                font-size:11px;">
                2. Review
            </span>

            <span style="color:#AAB3BF;">→</span>

            <span style="
                background:#EDF1F5;
                color:#68788B;
                padding:7px 13px;
                border-radius:20px;
                font-size:11px;">
                3. Results
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-card">
            <div class="step-label">STEP 01 · SELECT YOUR IMAGE</div>
            <div class="card-heading" style="margin-top:9px;">
                Upload an image
            </div>
            <div class="card-description">
                Choose a JPG, JPEG or PNG image from your device.
                Maximum file size: 10 MB.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=ALLOWED_EXTENSIONS,
        help="Supported formats: JPG, JPEG and PNG. Maximum size: 10 MB.",
        key="image_uploader",
    )

    if uploaded_file is not None:

        valid, message = validate_uploaded_image(uploaded_file)

        if not valid:
            st.error(message)

        else:

            uploaded_bytes = uploaded_file.getvalue()
            current_hash = get_image_hash(uploaded_bytes)

            if st.session_state.uploaded_hash != current_hash:
                st.session_state.uploaded_hash = current_hash

            try:
                original_image = Image.open(
                    BytesIO(uploaded_bytes)
                )

                original_image = ImageOps.exif_transpose(
                    original_image
                ).convert("RGB")

                st.write("")
                st.markdown(
                    '<div class="step-label">STEP 02 · REVIEW YOUR IMAGE</div>',
                    unsafe_allow_html=True,
                )

                image_col, details_col = st.columns(
                    [1.15, 1],
                    gap="large",
                )

                with image_col:

                    st.image(
                        original_image,
                        caption="Uploaded image preview",
                        use_container_width=True,
                    )

                with details_col:

                    st.markdown(
                        """
                        <div class="app-card">
                            <div class="card-heading">
                                Image details
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write("")

                    st.markdown(
                        f"**File:** {escape(uploaded_file.name)}"
                    )

                    st.markdown(
                        f"**Dimensions:** {original_image.width} × {original_image.height}"
                    )

                    st.markdown(
                        f"**Format:** {escape(original_image.format or 'RGB image')}"
                    )

                    st.markdown(
                        f"**Size:** {uploaded_file.size / 1024:.1f} KB"
                    )

                    st.markdown(
                        """
                        <div class="notice" style="margin-top:15px;">
                            Please review the image before starting
                            the model analysis.
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")

                if st.button(
                    "Analyze image  →",
                    use_container_width=True,
                    type="primary",
                ):

                    try:

                        with st.spinner(
                            "Preparing the model and analyzing your image..."
                        ):

                            model = load_model()

                            prediction = run_prediction(
                                original_image,
                                model,
                            )

                            analysis_id = get_image_hash(uploaded_bytes)[:12]

                            prediction["filename"] = uploaded_file.name
                            prediction["image_bytes"] = uploaded_bytes
                            prediction["analysis_id"] = analysis_id

                            st.session_state.current_analysis = prediction

                            st.session_state.history.insert(
                                0,
                                {
                                    "filename": uploaded_file.name,
                                    "result": prediction["result"],
                                    "confidence": prediction["confidence"],
                                    "timestamp": prediction["timestamp"],
                                    "analysis_id": analysis_id,
                                    "probabilities": prediction["probabilities"],
                                    "tensor_shape": prediction["tensor_shape"],
                                    "image_bytes": uploaded_bytes,
                                },
                            )

                            st.session_state.history = (
                                st.session_state.history[:20]
                            )

                        st.success("Analysis completed.")

                    except Exception as error:
                        st.error(
                            "The analysis could not be completed."
                        )
                        st.exception(error)

                if st.session_state.current_analysis is not None:

                    current = st.session_state.current_analysis

                    if current["analysis_id"] == current_hash[:12]:

                        st.write("")
                        st.divider()

                        st.markdown(
                            '<div class="step-label">STEP 03 · YOUR RESULTS</div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            '<div class="section-title">Analysis results</div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            """
                            <div class="section-description">
                                The following information represents
                                the output generated by the model.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.write("")

                        result_col, confidence_col = st.columns(2)

                        with result_col:

                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="result-label">
                                        MODEL OUTPUT
                                    </div>
                                    <div class="result-number">
                                        {escape(current["result"])}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with confidence_col:

                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="result-label">
                                        MODEL CONFIDENCE
                                    </div>
                                    <div class="result-number">
                                        {current["confidence"]:.2f}%
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.write("")
                        st.markdown("#### Probability breakdown")

                        for name, probability in zip(
                            CLASS_NAMES,
                            current["probabilities"],
                        ):

                            percentage = float(probability) * 100

                            st.markdown(
                                f"**{name}** · {percentage:.2f}%"
                            )

                            st.progress(
                                float(np.clip(probability, 0, 1))
                            )

                        st.caption(
                            "These are model output scores, not validated "
                            "medical probabilities."
                        )

                        st.write("")

                        st.markdown(
                            """
                            <div class="safety-notice">
                                <b>Please remember</b><br>
                                This output is experimental and educational.
                                It cannot establish whether a person has or
                                does not have cancer. Do not use it for
                                medical decisions.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.write("")

                        st.markdown(
                            '<div class="step-label">STEP 04 · SAVE YOUR REPORT</div>',
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "Generate PDF report",
                            use_container_width=True,
                        ):

                            try:

                                with st.spinner(
                                    "Creating your PDF report..."
                                ):

                                    model = load_model()

                                    report = generate_analysis_pdf(
                                        result=current,
                                        original_image=original_image,
                                        filename=current["filename"],
                                        model=model,
                                    )

                                st.download_button(
                                    label="Download professional PDF",
                                    data=report,
                                    file_name=(
                                        "BreastScan_AI_Report_"
                                        + datetime.now().strftime(
                                            "%Y%m%d_%H%M%S"
                                        )
                                        + ".pdf"
                                    ),
                                    mime="application/pdf",
                                    use_container_width=True,
                                )

                            except Exception as error:
                                st.error(
                                    "Could not generate the PDF report."
                                )
                                st.exception(error)

            except Exception as error:
                st.error("Unable to display the uploaded image.")
                st.exception(error)

    else:

        st.info(
            "Start by selecting an image using the upload area above."
        )

    show_footer()


# ============================================================
# 13. HISTORY PAGE
# ============================================================

elif st.session_state.page == "My history":

    show_page_heading(
        "YOUR WORKSPACE",
        "Analysis history",
        "Review analyses completed during this app session.",
    )

    if not st.session_state.history:

        st.markdown(
            """
            <div class="app-card" style="text-align:center;padding:45px 20px;">
                <div style="font-size:36px;">📂</div>
                <div style="
                    font-family:'Playfair Display',serif;
                    font-style:italic;
                    font-size:24px;
                    color:#263957;
                    margin-top:12px;">
                    Nothing here yet
                </div>
                <div class="card-description">
                    Your completed analyses will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start your first analysis"):
            navigate_to("New analysis")

    else:

        st.caption(
            f"{len(st.session_state.history)} recent analysis record(s)"
        )

        for index, item in enumerate(st.session_state.history):

            with st.container(border=True):

                col1, col2, col3 = st.columns(
                    [2, 1, 1],
                    vertical_alignment="center",
                )

                with col1:
                    st.markdown(
                        f"**{escape(item['filename'])}**"
                    )
                    st.caption(
                        item["timestamp"].strftime(
                            "%d %B %Y · %I:%M %p"
                        )
                    )

                with col2:
                    st.markdown(
                        f"**{escape(item['result'])}**"
                    )
                    st.caption("Model output")

                with col3:
                    st.markdown(
                        f"**{item['confidence']:.2f}%**"
                    )
                    st.caption("Model confidence")

                with st.expander("View analysis details"):

                    st.markdown("**Probability breakdown**")

                    for name, probability in zip(
                        CLASS_NAMES,
                        item["probabilities"],
                    ):

                        st.write(
                            f"{name}: {float(probability) * 100:.2f}%"
                        )

                    if st.button(
                        "Open this analysis",
                        key=f"history_{index}",
                    ):

                        st.session_state.current_analysis = item
                        navigate_to("New analysis")

        st.write("")

        # CSV export
        csv_rows = [
            "Filename,Model Output,Confidence,Timestamp"
        ]

        for item in st.session_state.history:

            safe_name = (
                str(item["filename"])
                .replace('"', '""')
                .replace("\n", " ")
            )

            csv_rows.append(
                f'"{safe_name}",'
                f'"{item["result"]}",'
                f'{item["confidence"]:.2f},'
                f'"{item["timestamp"].isoformat()}"'
            )

        csv_data = "\n".join(csv_rows)

        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="BreastScan_AI_History.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button(
            "Clear session history",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.session_state.current_analysis = None
            st.rerun()

        st.caption(
            "History is stored only in the current Streamlit session. "
            "It is not a permanent patient record."
        )

    show_footer()


# ============================================================
# 14. ABOUT PROJECT PAGE
# ============================================================

elif st.session_state.page == "About project":

    show_page_heading(
        "ABOUT THE PROJECT",
        "Technology with a purpose.",
        "Learn about the project, its workflow and its limitations.",
    )

    st.markdown(
        """
        <div class="app-card">
            <div class="eyebrow">OUR PROJECT</div>

            <div style="
                font-family:'Playfair Display',serif;
                font-style:italic;
                font-size:30px;
                color:#263957;
                margin-top:10px;">
                BreastScan AI
            </div>

            <div class="card-description">
                BreastScan AI is an academic machine-learning project
                demonstrating image preprocessing, model inference
                and an interactive Streamlit interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown("### Project overview")

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="app-card">
                <div class="card-heading">Image preprocessing</div>
                <div class="card-description">
                    The uploaded image is converted into the expected
                    color format, resized to the model input dimensions
                    and normalized before inference.
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
                    A trained Keras model processes the image and
                    returns output scores for the configured classes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown("### Technical information")

    technical_details = {
        "Application": "Streamlit",
        "Model framework": "TensorFlow / Keras",
        "Model file": "model.h5",
        "Input format": "RGB or grayscale, depending on model",
        "Image normalization": "Pixel values divided by 255",
        "Output classes": "Benign / Malignant",
        "Report generation": "ReportLab PDF",
    }

    for key, value in technical_details.items():

        col1, col2 = st.columns([1, 2])

        with col1:
            st.caption(key)

        with col2:
            st.write(value)

    st.divider()

    st.markdown("### Our team")

    st.markdown(
        f"""
        <div class="app-card">
            <div class="eyebrow">ACADEMIC PROJECT · {ACADEMIC_YEAR}</div>

            <div style="
                font-family:'Playfair Display',serif;
                font-style:italic;
                font-size:24px;
                color:#263957;
                margin-top:10px;">
                {escape(TEAM_NAME)}
            </div>

            <div class="card-description">
                Army Public School, LBS Marg, Lucknow
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
            validated diagnostic tool.

            <br><br>

            Model predictions may be affected by image quality,
            dataset limitations, training procedures and other
            technical factors.

            <br><br>

            A Benign output does not rule out cancer, and a Malignant
            output does not confirm cancer. Never use this application
            to make medical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_footer()
