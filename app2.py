"""Breast Scan AI — Streamlit interface for the histopathology classifier."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as _escape

import gdown
import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import Image as RLImage

MODEL_FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
MODEL_PATH = "model.h5"
LOGO_PATH = "logo.png"
BACKDROP_PATH = "bg.png"
VIDEO_PATH = "wlcmvid.mp4"

CLASSES = ("Benign", "Malignant")
DEMO_URL = "https://breast-cancer-classification-aps.streamlit.app/"
PROJECT_URL = "https://github.com/aiexploregroup-source/breast-cancer-classification-aps"
CONTACT_EMAIL = "aiexploregroup@gmail.com"

# ---------------------------------------------------------------------------
# PDF report generation (ReportLab)
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4
MARGIN = 42
TOP_MARGIN = 45
BOTTOM_MARGIN = 55
CONTENT_W = PAGE_W - 2 * MARGIN

BRAND = HexColor("#C2185B")
BRAND_DARK = HexColor("#8E1145")
BRAND_LIGHT = HexColor("#E23A7E")
INK = HexColor("#3A1427")
MUTED = HexColor("#8A5A70")
LINE = HexColor("#F3D3E1")
LINE_SOFT = HexColor("#F7E1EC")
BG_SOFT = HexColor("#FFF8FB")
PINK_SOFT = HexColor("#FDECF3")
WHITE = colors.white
SHADOW = HexColor("#F7DCE9")
TRACK = HexColor("#FBE4EE")

GREEN = HexColor("#0E9F6E")
GREEN_INK = HexColor("#0B7A55")
GREEN_BG = HexColor("#E8FBF3")
GREEN_LINE = HexColor("#B9EDD8")
RED = HexColor("#D7263D")
RED_INK = HexColor("#B0182C")
RED_BG = HexColor("#FDECEF")
RED_LINE = HexColor("#F5C3CB")

GREEN_INK_H = "#0B7A55"
RED_INK_H = "#B0182C"
BRAND_DARK_H = "#8E1145"

FONT = "Helvetica"
FONT_B = "Helvetica-Bold"
FONT_I = "Helvetica-Oblique"
FONT_BI = "Helvetica-BoldOblique"
_STYLES: dict | None = None
_LOGO_CACHE: ImageReader | None = None

_FONT_FAMILIES = (
    (
        "SegoeUI",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeuii.ttf",
        "C:/Windows/Fonts/segoeuiz.ttf",
    ),
    (
        "DejaVuSans",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    ),
    (
        "LiberationSans",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    ),
)


def _register_fonts() -> None:
    """Prefer a clean UI sans-serif; fall back to the built-in Helvetica."""
    global FONT, FONT_B, FONT_I, FONT_BI
    if FONT != "Helvetica":
        return
    for family, regular, bold, italic, bold_italic in _FONT_FAMILIES:
        if not (os.path.exists(regular) and os.path.exists(bold)):
            continue
        try:
            pdfmetrics.registerFont(TTFont(family, regular))
            pdfmetrics.registerFont(TTFont(family + "-Bold", bold))
            has_italic = os.path.exists(italic)
            has_bold_italic = os.path.exists(bold_italic)
            if has_italic:
                pdfmetrics.registerFont(TTFont(family + "-Italic", italic))
            if has_bold_italic:
                pdfmetrics.registerFont(TTFont(family + "-BoldItalic", bold_italic))
            pdfmetrics.registerFontFamily(
                family,
                normal=family,
                bold=family + "-Bold",
                italic=family + "-Italic" if has_italic else family,
                boldItalic=family + "-BoldItalic" if has_bold_italic else family + "-Bold",
            )
        except Exception:
            continue
        FONT = family
        FONT_B = family + "-Bold"
        FONT_I = family + "-Italic" if has_italic else family
        FONT_BI = family + "-BoldItalic" if has_bold_italic else family + "-Bold"
        return


def _styles() -> dict:
    global _STYLES
    if _STYLES is not None:
        return _STYLES
    _register_fonts()
    _STYLES = {
        "eyebrow": ParagraphStyle(
            "eyebrow", fontName=FONT_B, fontSize=7.4, leading=9, textColor=BRAND
        ),
        "h2": ParagraphStyle("h2", fontName=FONT_B, fontSize=13.2, leading=16, textColor=INK),
        "h3": ParagraphStyle("h3", fontName=FONT_B, fontSize=11, leading=14, textColor=INK),
        "body": ParagraphStyle("body", fontName=FONT, fontSize=8.9, leading=13.2, textColor=INK),
        "small": ParagraphStyle("small", fontName=FONT, fontSize=8.2, leading=11.8, textColor=MUTED),
        "tiny": ParagraphStyle("tiny", fontName=FONT, fontSize=7.4, leading=10.4, textColor=MUTED),
        "link": ParagraphStyle("link", fontName=FONT, fontSize=7.4, leading=10.4, textColor=BRAND),
        "key": ParagraphStyle("key", fontName=FONT_B, fontSize=7.6, leading=10.6, textColor=MUTED),
        "value": ParagraphStyle("value", fontName=FONT, fontSize=8.3, leading=11.2, textColor=INK),
        "filename": ParagraphStyle("filename", fontName=FONT_B, fontSize=8.4, leading=11, textColor=INK),
        "step_num": ParagraphStyle("step_num", fontName=FONT_B, fontSize=15, leading=17, textColor=BRAND),
        "step_title": ParagraphStyle("step_title", fontName=FONT_B, fontSize=9.6, leading=12, textColor=INK),
        "step_body": ParagraphStyle("step_body", fontName=FONT, fontSize=8, leading=11.4, textColor=MUTED),
        "bullet": ParagraphStyle(
            "bullet",
            fontName=FONT,
            fontSize=8.5,
            leading=12.4,
            textColor=INK,
            leftIndent=11,
            bulletIndent=0,
            bulletFontName=FONT_B,
            bulletFontSize=8.5,
        ),
        "verdict_eyebrow": ParagraphStyle(
            "verdict_eyebrow", fontName=FONT_B, fontSize=7.2, leading=8.8, textColor=MUTED
        ),
        "verdict_title": ParagraphStyle(
            "verdict_title", fontName=FONT_B, fontSize=16.5, leading=19.5, textColor=INK
        ),
        "verdict_sub": ParagraphStyle(
            "verdict_sub", fontName=FONT, fontSize=8.2, leading=11.2, textColor=MUTED
        ),
    }
    return _STYLES


def _panel(content, *, width=None, min_height=0, bg=WHITE, border=LINE, radius=14,
           padding=14, shadow=True, accent=None) -> "Flowable":
    return _Panel(content, width=width, min_height=min_height, bg=bg, border=border,
                  radius=radius, padding=padding, shadow=shadow, accent=accent)


class _Panel(Flowable):
    """Rounded card that lays out arbitrary flowables inside itself."""

    def __init__(self, content, *, width=None, min_height=0, bg=WHITE, border=LINE,
                 radius=14, padding=14, shadow=True, accent=None):
        super().__init__()
        self._content = list(content)
        self._fixed_w = width
        self._min_h = min_height
        self.bg = bg
        self.border = border
        self.radius = radius
        self.padding = padding
        self.shadow = shadow
        self.accent = accent
        self._heights: list[float] = []

    def wrap(self, avail_w, avail_h):
        self.width = self._fixed_w or avail_w
        inner = self.width - 2 * self.padding
        total = 0.0
        self._heights = []
        for flow in self._content:
            _, height = flow.wrap(inner, avail_h)
            self._heights.append(height)
            total += height
        self.height = max(self._min_h, total + 2 * self.padding)
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        if self.shadow:
            c.setFillColor(SHADOW)
            c.roundRect(0, -2.4, self.width, self.height, self.radius, stroke=0, fill=1)
        c.setFillColor(self.bg)
        c.setStrokeColor(self.border)
        c.setLineWidth(0.8)
        c.roundRect(0, 0, self.width, self.height, self.radius, stroke=1, fill=1)
        if self.accent is not None:
            c.setFillColor(self.accent)
            c.roundRect(0, 12, 3.4, self.height - 24, 1.7, stroke=0, fill=1)
        y = self.height - self.padding
        inner = self.width - 2 * self.padding
        for flow, height in zip(self._content, self._heights):
            y -= height
            x = self.padding
            if isinstance(flow, RLImage):
                draw_w = getattr(flow, "drawWidth", inner)
                if draw_w < inner:
                    x += (inner - draw_w) / 2
            flow.drawOn(c, x, y)
        c.restoreState()


class _SectionHeading(Flowable):
    def __init__(self, title, eyebrow=None, styles=None):
        super().__init__()
        styles = styles or _styles()
        self._title = Paragraph(_escape(title), styles["h2"])
        self._eyebrow = Paragraph(_escape(eyebrow).upper(), styles["eyebrow"]) if eyebrow else None

    def wrap(self, avail_w, avail_h):
        self.width = avail_w
        height = 0.0
        if self._eyebrow is not None:
            height += self._eyebrow.wrap(avail_w, avail_h)[1] + 1
        height += self._title.wrap(avail_w, avail_h)[1] + 7
        self.height = height
        return self.width, self.height

    def draw(self):
        c = self.canv
        y = self.height
        if self._eyebrow is not None:
            y -= self._eyebrow.height
            self._eyebrow.drawOn(c, 0, y)
            y -= 1
        y -= self._title.height
        self._title.drawOn(c, 0, y)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.7)
        c.line(0, 0.4, self.width, 0.4)


class _TitleBand(Flowable):
    def __init__(self, logo, meta_line, height=102):
        super().__init__()
        self._logo = logo
        self._meta = meta_line
        self._height = height

    def wrap(self, avail_w, avail_h):
        self.width = avail_w
        self.height = self._height
        return self.width, self.height

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        c.saveState()
        clip = c.beginPath()
        clip.roundRect(0, 0, w, h, 20)
        c.clipPath(clip, stroke=0, fill=0)
        c.linearGradient(
            0, h, w, 0, [BRAND_DARK, BRAND, BRAND_LIGHT], positions=[0, 0.58, 1], extend=True
        )
        c.setFillColor(WHITE)
        c.setFillAlpha(0.09)
        c.circle(w - 58, h - 16, 82, stroke=0, fill=1)
        c.circle(w - 6, 2, 44, stroke=0, fill=1)
        c.setFillAlpha(1)
        c.restoreState()

        cx, cy, r = 54, h / 2, 31
        c.setFillColor(WHITE)
        c.circle(cx, cy, r, stroke=0, fill=1)
        if self._logo is not None:
            c.drawImage(self._logo, cx - 21, cy - 21, 42, 42)
        else:
            c.setFillColor(BRAND)
            c.setFont(FONT_B, 17)
            c.drawCentredString(cx, cy - 6.5, "AI")

        x = 98
        c.setFillColor(WHITE)
        c.setFont(FONT_B, 20.5)
        c.drawString(x, cy + 12, "Breast Scan AI")
        c.setFillAlpha(0.86)
        c.setFont(FONT, 9.4)
        c.drawString(x, cy - 5, "Histopathology analysis report")
        c.setFillAlpha(0.68)
        c.setFont(FONT, 7.5)
        c.drawString(x, cy - 21, self._meta)
        c.setFillAlpha(1)


class _VerdictBanner(Flowable):
    def __init__(self, benign: bool, score: float, styles=None):
        super().__init__()
        self.benign = benign
        self.score = score
        self.accent = GREEN if benign else RED
        self.bg = GREEN_BG if benign else RED_BG
        self.border = GREEN_LINE if benign else RED_LINE
        self.ink = GREEN_INK if benign else RED_INK
        ink_hex = GREEN_INK_H if benign else RED_INK_H
        styles = styles or _styles()
        title = "Benign pattern" if benign else "Malignant pattern"
        sub = (
            "The model assigned the highest score to benign tissue."
            if benign
            else "The model assigned the highest score to malignant tissue."
        )
        self._eyebrow = Paragraph(
            f'<font color="{ink_hex}">MODEL VERDICT</font>', styles["verdict_eyebrow"]
        )
        self._title = Paragraph(f'<font color="{ink_hex}">{title}</font>', styles["verdict_title"])
        self._sub = Paragraph(_escape(sub), styles["verdict_sub"])
        self._heights = [0.0, 0.0, 0.0]

    def wrap(self, avail_w, avail_h):
        self.width = avail_w
        text_w = avail_w - 76 - 148
        self._heights = [
            self._eyebrow.wrap(text_w, avail_h)[1],
            self._title.wrap(text_w, avail_h)[1],
            self._sub.wrap(text_w, avail_h)[1],
        ]
        block = sum(self._heights) + 8
        self.height = max(86, block + 26)
        return self.width, self.height

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        c.saveState()
        c.setFillColor(self.bg)
        c.setStrokeColor(self.border)
        c.setLineWidth(0.9)
        c.roundRect(0, 0, w, h, 14, stroke=1, fill=1)

        icon_x, icon_y, icon_r = 40, h / 2, 20
        c.setFillColor(self.accent)
        c.circle(icon_x, icon_y, icon_r, stroke=0, fill=1)
        c.setStrokeColor(WHITE)
        c.setFillColor(WHITE)
        c.setLineWidth(3)
        c.setLineCap(1)
        c.setLineJoin(1)
        if self.benign:
            p = c.beginPath()
            p.moveTo(icon_x - 8, icon_y + 0.5)
            p.lineTo(icon_x - 2.5, icon_y - 5.5)
            p.lineTo(icon_x + 8.5, icon_y + 6)
            c.drawPath(p, stroke=1, fill=0)
        else:
            c.setLineWidth(3.6)
            c.line(icon_x, icon_y + 8, icon_x, icon_y - 1)
            c.circle(icon_x, icon_y - 8, 1.9, stroke=0, fill=1)
        c.setLineCap(0)
        c.setLineJoin(0)

        block = sum(self._heights) + 8
        y = (h + block) / 2
        for para, gap in zip((self._eyebrow, self._title, self._sub), (2, 2, 0)):
            y -= para.height
            para.drawOn(c, 76, y)
            y -= gap

        c.setStrokeColor(self.border)
        c.setLineWidth(0.7)
        c.line(w - 138, h / 2 - 19, w - 138, h / 2 + 19)
        c.setFillColor(self.accent)
        c.setFont(FONT_B, 27)
        c.drawRightString(w - 20, h / 2 - 2, f"{self.score:.1%}")
        c.setFillColor(MUTED)
        c.setFont(FONT, 7.2)
        c.drawRightString(w - 20, h / 2 - 17, "model score")
        c.restoreState()


def _draw_arc(c, cx, cy, r, start_deg, end_deg, width, color, steps=64):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setLineCap(1)
    c.setLineJoin(1)
    path = c.beginPath()
    for i in range(steps + 1):
        angle = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()


class _Gauge(Flowable):
    def __init__(self, value: float, accent, width=212, height=136):
        super().__init__()
        self.value = max(0.0, min(1.0, float(value)))
        self.accent = accent
        self._w = width
        self._h = height

    def wrap(self, avail_w, avail_h):
        self.width = min(self._w, avail_w)
        self.height = self._h
        return self.width, self.height

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        cx = w / 2
        cy = 38
        r = 72
        _draw_arc(c, cx, cy, r, 0, 180, 12, TRACK)

        for tick in range(0, 101, 10):
            angle = math.radians(180 - 1.8 * tick)
            outer = r + 10
            inner = r + 5.5
            c.setStrokeColor(LINE)
            c.setLineWidth(0.9)
            c.line(
                cx + inner * math.cos(angle),
                cy + inner * math.sin(angle),
                cx + outer * math.cos(angle),
                cy + outer * math.sin(angle),
            )

        if self.value > 0:
            _draw_arc(c, cx, cy, r, 180, 180 - 180 * self.value, 12, self.accent)
            end_angle = math.radians(180 - 180 * self.value)
            c.setFillColor(self.accent)
            c.circle(cx + r * math.cos(end_angle), cy + r * math.sin(end_angle), 6, stroke=0, fill=1)

        c.setFillColor(INK)
        c.setFont(FONT_B, 25)
        c.drawCentredString(cx, cy + 20, f"{self.value:.1%}")
        c.setFillColor(MUTED)
        c.setFont(FONT, 7.2)
        c.drawCentredString(cx, cy + 6, "model score")
        c.setFont(FONT, 6.8)
        c.setFillColor(MUTED)
        c.drawCentredString(cx - r, cy - 13, "0%")
        c.drawCentredString(cx + r, cy - 13, "100%")


class _ScoreBars(Flowable):
    def __init__(self, rows, width=247, height=64):
        super().__init__()
        self._rows = rows
        self._w = width
        self._h = height

    def wrap(self, avail_w, avail_h):
        self.width = min(self._w, avail_w)
        self.height = self._h
        return self.width, self.height

    def draw(self):
        c = self.canv
        bar_x = 72
        bar_w = self.width - bar_x - 48
        for i, (label, value, color) in enumerate(self._rows):
            cy = self.height - 12 - i * 30
            c.setFillColor(INK)
            c.setFont(FONT_B, 8.8)
            c.drawString(0, cy - 3.2, label)
            c.setFillColor(TRACK)
            c.roundRect(bar_x, cy - 4.5, bar_w, 9, 4.5, stroke=0, fill=1)
            c.setFillColor(color)
            c.roundRect(bar_x, cy - 4.5, max(9, bar_w * value), 9, 4.5, stroke=0, fill=1)
            c.setFillColor(color)
            c.setFont(FONT_B, 9.4)
            c.drawRightString(self.width, cy - 3.4, f"{value:.1%}")


def _kv_table(rows, width, styles, key_ratio=0.4):
    data = [[Paragraph(_escape(str(k)), styles["key"]), Paragraph(v, styles["value"])] for k, v in rows]
    table = Table(data, colWidths=[width * key_ratio, width * (1 - key_ratio)], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.4),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE_SOFT),
            ]
        )
    )
    return table


def _image_flowable(image: Image.Image, box_w: float, box_h: float) -> RLImage:
    picture = image.convert("RGB")
    width, height = picture.size
    scale = min(box_w / width, box_h / height)
    disp_w, disp_h = width * scale, height * scale
    picture.thumbnail((900, 900), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=88, optimize=True)
    buffer.seek(0)
    flow = RLImage(buffer, width=disp_w, height=disp_h)
    flow._bsa_buffer = buffer  # keep the stream alive while the document builds
    return flow


def _logo_reader() -> ImageReader | None:
    global _LOGO_CACHE
    if _LOGO_CACHE is not None:
        return _LOGO_CACHE
    path = Path(__file__).resolve().parent / "logo.png"
    if not path.exists():
        return None
    try:
        logo = Image.open(path).convert("RGBA")
        canvas_img = Image.new("RGB", logo.size, "white")
        canvas_img.paste(logo, mask=logo.split()[-1])
        canvas_img.thumbnail((320, 320), Image.LANCZOS)
        buffer = io.BytesIO()
        canvas_img.save(buffer, format="PNG")
        buffer.seek(0)
        _LOGO_CACHE = ImageReader(buffer)
    except Exception:
        return None
    return _LOGO_CACHE


def _report_id(result: dict, generated: datetime) -> str:
    seed = "|".join(
        [
            str(result.get("filename", "")),
            str(result.get("benign", "")),
            str(result.get("malignant", "")),
            generated.isoformat(timespec="seconds"),
        ]
    )
    return "BSA-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8].upper()


def _fmt_size(kilobytes: float) -> str:
    if kilobytes >= 1024:
        return f"{kilobytes / 1024:.1f} MB"
    return f"{kilobytes:.0f} KB"


def _short_filename(name: str, limit: int = 46) -> str:
    clean = "".join(ch for ch in name if ch.isprintable()).strip()
    if len(clean) <= limit:
        return clean
    stem, ext = os.path.splitext(clean)
    if ext and len(ext) <= 8:
        tail = stem[-10:]
        head = limit - len(ext) - 1 - len(tail) - 1
        if head >= 8:
            return stem[:head].rstrip() + "…" + tail + ext
    return clean[: limit - 1].rstrip() + "…"


def _step_card(number: str, title: str, body: str, styles, width: float) -> _Panel:
    return _panel(
        [
            Paragraph(number, styles["step_num"]),
            Spacer(1, 3),
            Paragraph(_escape(title), styles["step_title"]),
            Spacer(1, 3),
            Paragraph(_escape(body), styles["step_body"]),
        ],
        width=width,
        min_height=96,
        padding=13,
        radius=12,
    )


class _NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        pdfcanvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_footer(self, total: int):
        c = self
        c.saveState()
        c.setStrokeColor(LINE)
        c.setLineWidth(0.7)
        c.line(MARGIN, BOTTOM_MARGIN - 15, PAGE_W - MARGIN, BOTTOM_MARGIN - 15)
        c.setFillColor(MUTED)
        c.setFont(FONT, 7.2)
        c.drawString(
            MARGIN,
            BOTTOM_MARGIN - 27,
            "Breast Scan AI  ·  Educational demonstration  ·  Not a medical device",
        )
        c.drawRightString(PAGE_W - MARGIN, BOTTOM_MARGIN - 27, f"Page {self._pageNumber} of {total}")
        c.restoreState()


def _draw_header(c, doc, report_id: str):
    if doc.page == 1:
        return
    c.saveState()
    c.setFillColor(INK)
    c.setFont(FONT_B, 8.4)
    c.drawString(MARGIN, PAGE_H - 29, "Breast Scan AI")
    c.setFillColor(MUTED)
    c.setFont(FONT, 8.4)
    offset = pdfmetrics.stringWidth("Breast Scan AI  ", FONT_B, 8.4)
    c.drawString(MARGIN + offset, PAGE_H - 29, "·  Histopathology analysis report")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 29, report_id)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.7)
    c.line(MARGIN, PAGE_H - 37, PAGE_W - MARGIN, PAGE_H - 37)
    c.restoreState()


def build_report(result: dict, image: Image.Image, generated: datetime | None = None) -> bytes:
    """Build the analysis PDF and return it as bytes."""
    styles = _styles()
    generated = generated or datetime.now()
    date_long = generated.strftime("%d %B %Y")
    date_short = generated.strftime("%d %b %Y")
    time_short = generated.strftime("%H:%M")
    report_id = _report_id(result, generated)

    benign = result["verdict"] == "Benign"
    accent = GREEN if benign else RED
    score = float(result["score"])
    benign_score = float(result["benign"])
    malignant_score = float(result["malignant"])

    logo = _logo_reader()
    band = _TitleBand(logo, f"Report {report_id}   ·   {date_long}   ·   {time_short}")

    gap = 14.0
    card_w = (CONTENT_W - gap) / 2

    sample_img = _image_flowable(image, card_w - 2 * 14, 112)
    sample_card = _panel(
        [
            Paragraph("SAMPLE", styles["eyebrow"]),
            Spacer(1, 2),
            Paragraph("Uploaded image", styles["h3"]),
            Spacer(1, 7),
            sample_img,
            Spacer(1, 7),
            Paragraph(_escape(_short_filename(result["filename"])), styles["filename"]),
            Spacer(1, 1),
            Paragraph(
                f"{result['width']} × {result['height']} px  ·  {result['format']}  ·  "
                f"{_fmt_size(float(result['kilobytes']))}",
                styles["tiny"],
            ),
        ],
        width=card_w,
        min_height=212,
    )

    verdict_html = (
        f'<font color="{GREEN_INK_H}"><b>Benign</b></font>'
        if benign
        else f'<font color="{RED_INK_H}"><b>Malignant</b></font>'
    )
    summary_rows = [
        ("Verdict", verdict_html),
        ("Model score", f"{score:.1%}"),
        ("Architecture", "Custom CNN"),
        ("Input resolution", "48 × 48 px RGB"),
        ("Output classes", "Benign · Malignant"),
        ("Framework", "TensorFlow / Keras"),
        ("Reported accuracy", "~92% (training)"),
    ]
    summary_card = _panel(
        [
            Paragraph("MODEL", styles["eyebrow"]),
            Spacer(1, 2),
            Paragraph("Analysis summary", styles["h3"]),
            Spacer(1, 8),
            _kv_table(summary_rows, card_w - 2 * 14, styles, key_ratio=0.42),
            Spacer(1, 7),
            Paragraph(
                f"Raw softmax — benign {benign_score:.4f}  ·  malignant {malignant_score:.4f}",
                styles["tiny"],
            ),
        ],
        width=card_w,
        min_height=212,
    )

    cards = Table(
        [[sample_card, "", summary_card]],
        colWidths=[card_w, gap, card_w],
        hAlign="LEFT",
    )
    cards.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    bars = _ScoreBars(
        [("Benign", benign_score, GREEN), ("Malignant", malignant_score, RED)],
        width=CONTENT_W - 2 * 14 - 218 - 14,
    )
    score_table = Table(
        [
            [
                _Gauge(score, accent),
                [
                    Paragraph("CLASS SCORES", styles["eyebrow"]),
                    Spacer(1, 6),
                    bars,
                    Spacer(1, 9),
                    Paragraph(
                        "Scores are raw softmax outputs from the model, not calibrated "
                        "probabilities. The verdict reflects the highest-scoring class.",
                        styles["tiny"],
                    ),
                ],
            ]
        ],
        colWidths=[218, CONTENT_W - 2 * 14 - 218 - 14],
        hAlign="LEFT",
    )
    score_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    score_panel = _panel([score_table], padding=14, radius=14)

    disclaimer = _panel(
        [
            Paragraph(
                f'<font color="{BRAND_DARK_H}"><b>Educational demonstration only.</b></font> '
                "Model outputs are not a diagnosis. Discuss any health concerns with a "
                "qualified clinician.",
                styles["small"],
            )
        ],
        bg=PINK_SOFT,
        border=LINE,
        padding=12,
        radius=12,
        shadow=False,
    )

    step_w = (CONTENT_W - 2 * 12) / 3
    steps = Table(
        [
            [
                _step_card("01", "Upload", "A single JPG or PNG histopathology crop is provided by the user.", styles, step_w),
                "",
                _step_card("02", "Preprocess", "The image is converted to RGB, resized to 48 × 48 px and scaled to 0–1.", styles, step_w),
                "",
                _step_card("03", "Classify", "The CNN outputs a two-class softmax distribution over benign and malignant.", styles, step_w),
            ]
        ],
        colWidths=[step_w, 12, step_w, 12, step_w],
        hAlign="LEFT",
    )
    steps.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    details_rows = [
        ("File name", _escape(_short_filename(result["filename"], 64))),
        ("Format", str(result["format"])),
        ("File size", _fmt_size(float(result["kilobytes"]))),
        ("Original dimensions", f"{result['width']} × {result['height']} px"),
        ("Model input", _escape(str(result["input_shape"]))),
        ("Class order", "0 = benign  ·  1 = malignant"),
        ("Raw scores", f"benign {benign_score:.4f}  ·  malignant {malignant_score:.4f}"),
        ("Generated", f"{date_long} at {time_short}"),
    ]
    details_panel = _panel(
        [_kv_table(details_rows, CONTENT_W - 2 * 14, styles, key_ratio=0.27)],
        padding=14,
        radius=14,
    )

    limitations_panel = _panel(
        [
            Paragraph(
                "This report reflects a single forward pass of the model on one uploaded "
                "image. It does not estimate uncertainty across images and cannot see tissue "
                "outside the uploaded crop.",
                styles["body"],
            ),
            Spacer(1, 7),
            Paragraph("Model scores are raw softmax outputs, not calibrated probabilities.", styles["bullet"], bulletText="•"),
            Spacer(1, 2),
            Paragraph("Reported accuracy reflects the team's training results, not independent clinical validation.", styles["bullet"], bulletText="•"),
            Spacer(1, 2),
            Paragraph("The verdict describes an image category, not a person's health.", styles["bullet"], bulletText="•"),
            Spacer(1, 2),
            Paragraph("Predictions must not guide screening, diagnosis or treatment decisions.", styles["bullet"], bulletText="•"),
        ],
        padding=14,
        radius=14,
    )

    final_disclaimer = _panel(
        [
            Paragraph(
                f'<font color="{BRAND_DARK_H}"><b>Not a medical device</b></font>',
                styles["h3"],
            ),
            Spacer(1, 4),
            Paragraph(
                "Breast Scan AI is an educational and research demonstration built on the "
                "public BreaKHis dataset. This document is not a medical record and carries "
                "no diagnostic authority. It must not be used for screening, treatment "
                "decisions, or reassurance about any person's health.",
                styles["small"],
            ),
            Spacer(1, 5),
            Paragraph(
                "Team Breast Scan AI  ·  Army Public School, LBS Marg, Lucknow  ·  "
                '<link href="https://github.com/aiexploregroup-source/breast-cancer-classification-aps">'
                "github.com/aiexploregroup-source/breast-cancer-classification-aps</link>",
                styles["link"],
            ),
        ],
        padding=14,
        radius=14,
        accent=BRAND,
        shadow=False,
        bg=WHITE,
    )

    signature = Table(
        [
            [
                Paragraph(
                    f"Report {report_id}  ·  Generated {date_short} at {time_short}",
                    styles["tiny"],
                ),
                Paragraph(
                    "Breast Scan AI  ·  Histopathology classifier",
                    ParagraphStyle(
                        "sig", parent=styles["tiny"], alignment=2, textColor=MUTED
                    ),
                ),
            ]
        ],
        colWidths=[CONTENT_W * 0.62, CONTENT_W * 0.38],
        hAlign="LEFT",
    )
    signature.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story = [
        band,
        Spacer(1, 14),
        _VerdictBanner(benign, score, styles),
        Spacer(1, 16),
        cards,
        Spacer(1, 18),
        KeepTogether(
            [
                _SectionHeading("Score breakdown", "Model output", styles),
                Spacer(1, 8),
                score_panel,
            ]
        ),
        Spacer(1, 16),
        disclaimer,
        PageBreak(),
        _SectionHeading("Methodology", "How this result was produced", styles),
        Spacer(1, 6),
        Paragraph(
            "The report documents a single prediction produced by the Breast Scan AI "
            "classifier from one uploaded image. No manual review or additional imaging "
            "was involved.",
            styles["small"],
        ),
        Spacer(1, 10),
        steps,
        Spacer(1, 14),
        KeepTogether(
            [
                _SectionHeading("File & processing details", "Sample data", styles),
                Spacer(1, 8),
                details_panel,
            ]
        ),
        Spacer(1, 14),
        KeepTogether(
            [
                _SectionHeading("Limitations & responsible use", "Read before use", styles),
                Spacer(1, 8),
                limitations_panel,
            ]
        ),
        Spacer(1, 12),
        final_disclaimer,
        Spacer(1, 8),
        signature,
    ]

    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=f"Breast Scan AI — Analysis Report {report_id}",
        author="Breast Scan AI",
        subject="Histopathology image classification report",
        creator="Breast Scan AI",
        keywords="breast cancer, histopathology, CNN, educational",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="report",
                frames=[frame],
                onPage=lambda c, d: _draw_header(c, d, report_id),
            )
        ]
    )
    doc.build(story, canvasmaker=_NumberedCanvas)
    return buffer.getvalue()


st.set_page_config(
    page_title="Breast Scan AI",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html(
    """
    <style>
      .st-key-hero {
        background:
          radial-gradient(640px 340px at 88% 4%, rgba(244, 143, 177, .18), rgba(244, 143, 177, 0) 70%),
          linear-gradient(118deg, #221018 0%, #2A1220 52%, #351626 100%);
        border: 1px solid #3E2332;
        border-radius: 28px;
        padding: 2.5rem 2.75rem;
        box-shadow: 0 32px 70px -40px rgba(0, 0, 0, .85);
      }
      .st-key-hero_logo {
        max-width: 320px;
        margin: 0 auto;
      }
      .st-key-hero_logo img {
        background: #FFFFFF;
        border-radius: 28px;
        padding: 14px;
        filter: drop-shadow(0 18px 30px rgba(216, 27, 96, .38));
        animation: bsa-float 7s ease-in-out infinite;
      }
      @keyframes bsa-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-9px); }
      }
      @media (prefers-reduced-motion: reduce) {
        .st-key-hero_logo img { animation: none; }
      }
      @media (max-width: 640px) {
        .st-key-hero { padding: 1.6rem 1.4rem; border-radius: 22px; }
      }

      .st-key-topbar { margin-bottom: 1.1rem; }
      .st-key-topbar img {
        border-radius: 12px;
        box-shadow: 0 6px 18px -8px rgba(0, 0, 0, .8);
      }
      .st-key-footer {
        margin-top: .5rem;
        border-top: 1px solid #3A2230;
        padding-top: .9rem;
      }

      .st-key-uploader [data-testid="stFileUploaderDropzone"] {
        background: #1B0F16;
        border: 1.5px dashed #5C2F46;
        border-radius: 18px;
        padding: 1.1rem;
      }
      .st-key-uploader [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #E4548A;
        background: #241320;
      }

      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stDecoration"] { display: none; }
    </style>
    """
)

st.session_state.setdefault("page", "Home")
st.session_state.setdefault("result", None)
st.session_state.setdefault("uploader_round", 0)
st.session_state.setdefault("report_cache", None)
st.session_state.setdefault("report_error", None)


def go_to(page_name: str) -> None:
    st.session_state.page = page_name


def reset_analysis() -> None:
    st.session_state.result = None
    st.session_state.report_cache = None
    st.session_state.report_error = None
    st.session_state.uploader_round += 1


@st.cache_resource(show_spinner=False)
def load_model():
    # Imported lazily so the home and about pages paint without waiting for TensorFlow.
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        if gdown.download(MODEL_URL, MODEL_PATH, quiet=True) is None:
            raise RuntimeError("Model download failed. Check the network connection and the Drive link.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def get_model():
    try:
        with st.spinner("Loading the classifier. The first analysis can take a minute."):
            return load_model()
    except Exception as exc:
        st.error(
            "The classifier could not be loaded. Check the network connection or the local model file.",
            title="Model unavailable",
            icon=":material/error:",
        )
        st.caption(f"Details: {exc}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").resize((48, 48))
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.expand_dims(pixels, axis=0)


def render_footer() -> None:
    st.space("large")
    with st.container(key="footer"):
        st.caption(
            "© 2026 Team Breast Scan AI · Army Public School, LBS Marg, Lucknow · "
            f"[GitHub]({PROJECT_URL}) · [Email](mailto:{CONTACT_EMAIL})",
            text_alignment="center",
        )
        st.caption("Educational demonstration only — not a medical device.", text_alignment="center")


def report_bytes(result: dict, image: Image.Image) -> bytes | None:
    cache = st.session_state.report_cache
    if cache is not None and cache[0] == result["signature"]:
        return cache[1]
    try:
        data = build_report(result, image)
    except Exception as exc:
        st.session_state.report_error = str(exc)
        return None
    st.session_state.report_cache = (result["signature"], data)
    st.session_state.report_error = None
    return data


def render_result(result: dict | None, image: Image.Image | None) -> None:
    has_image = image is not None
    if result is None:
        with st.container(border=True, height="stretch", horizontal_alignment="center"):
            st.markdown("### :material/biotech:")
            st.markdown("**Ready when you are**" if has_image else "**No analysis yet**")
            st.caption(
                "Choose **Run analysis** under the image to score it."
                if has_image
                else "Upload an image on the left to begin."
            )
        return

    benign = result["verdict"] == "Benign"
    if benign:
        st.success(
            "This image received the model's highest score for benign tissue.",
            title="Benign pattern",
            icon=":material/check_circle:",
        )
    else:
        st.error(
            "This image received the model's highest score for malignant tissue.",
            title="Malignant pattern",
            icon=":material/error:",
        )

    st.metric("Model score", f"{result['score']:.1%}", border=True)
    st.markdown("**Class scores**")
    st.progress(result["benign"], text=f"Benign — {result['benign']:.1%}")
    st.progress(result["malignant"], text=f"Malignant — {result['malignant']:.1%}")
    st.caption("Scores are raw softmax outputs from the model, not calibrated probabilities.")

    st.warning(
        "A model score is not a diagnosis. Discuss any health concerns with a qualified clinician.",
        title="Educational use only",
        icon=":material/health_and_safety:",
    )

    safe_name = result["filename"].replace("`", "'")
    with st.expander("Analysis details", icon=":material/troubleshoot:"):
        st.markdown(
            f"- **File:** `{safe_name}`\n"
            f"- **Original size:** {result['width']} × {result['height']} px\n"
            f"- **Format:** {result['format']}\n"
            f"- **File size:** {result['kilobytes']:.0f} KB\n"
            f"- **Model input:** {result['input_shape']}\n"
            "- **Class order:** 0 = benign, 1 = malignant\n"
            f"- **Raw scores:** benign {result['benign']:.4f}, malignant {result['malignant']:.4f}"
        )

    if image is not None:
        data = report_bytes(result, image)
        if data is not None:
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(result["filename"])[0]).strip("_")[:40] or "scan"
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            st.download_button(
                "Download PDF report",
                data=data,
                file_name=f"BreastScanAI_{stem}_{stamp}.pdf",
                mime="application/pdf",
                icon=":material/picture_as_pdf:",
                width="stretch",
            )
            st.caption("Two-page A4 report with the image, scores, methodology and disclaimers.")
        else:
            st.caption("The PDF report could not be generated for this analysis.")
            if st.session_state.report_error:
                with st.expander("Technical details", icon=":material/bug_report:"):
                    st.code(st.session_state.report_error)


with st.container(key="topbar"):
    brand_col, nav_col = st.columns([1.7, 1], vertical_alignment="center", gap="large")
    with brand_col:
        logo_col, name_col = st.columns([0.13, 0.87], vertical_alignment="center", gap="small")
        with logo_col:
            st.image(LOGO_PATH, width=44, alt="Breast Scan AI logo")
        with name_col:
            st.markdown("**Breast Scan AI**")
            st.caption("Histopathology classifier")
    with nav_col:
        page = st.segmented_control(
            "Navigate",
            ["Home", "Analyze", "About"],
            key="page",
            required=True,
            label_visibility="collapsed",
            width="stretch",
        )

if page == "Home":
    with st.container(key="hero"):
        intro_col, art_col = st.columns([1.3, 0.7], vertical_alignment="center", gap="large")
        with intro_col:
            st.badge("Educational research demo", icon=":material/school:", color="violet")
            st.title("Clarity for every scan.")
            st.markdown(
                "Breast Scan AI classifies breast histopathology images as **benign** or "
                "**malignant** using a convolutional neural network trained on the BreaKHis dataset."
            )
            cta_col, secondary_col = st.columns(2, gap="small")
            with cta_col:
                st.button(
                    "Analyze an image",
                    type="primary",
                    icon=":material/upload_file:",
                    on_click=go_to,
                    args=("Analyze",),
                    width="stretch",
                )
            with secondary_col:
                st.button(
                    "About the project",
                    icon=":material/info:",
                    on_click=go_to,
                    args=("About",),
                    width="stretch",
                )
            st.caption("For education and research. Not a medical diagnosis.")
        with art_col:
            with st.container(key="hero_logo"):
                st.image(
                    LOGO_PATH,
                    width="stretch",
                    alt="Pink awareness ribbon wrapped around an AI badge",
                )

    st.space("medium")
    stat_cols = st.columns(3, border=True)
    stats = (
        (":material/straighten:", "48 × 48 px", "Model input resolution"),
        (":material/category:", "2 classes", "Benign and malignant"),
        (":material/memory:", "Custom CNN", "Built with TensorFlow / Keras"),
    )
    for col, (icon, value, label) in zip(stat_cols, stats):
        with col:
            st.markdown(icon)
            st.markdown(f"### {value}")
            st.caption(label)

    st.space("medium")
    st.subheader("How it works", icon=":material/route:")
    st.caption("Three steps from image to model output.")
    step_cols = st.columns(3, border=True)
    steps = (
        ("Step 1", "Upload", "Add a single JPG or PNG histopathology image."),
        ("Step 2", "Analyze", "The image is converted to RGB, resized to 48 × 48 px, and scored by the CNN."),
        ("Step 3", "Review", "Read the predicted class and both class scores, then discuss with a clinician."),
    )
    for col, (badge_label, title, body) in zip(step_cols, steps):
        with col:
            st.badge(badge_label, color="gray")
            st.markdown(f"#### {title}")
            st.caption(body)

    st.space("medium")
    st.subheader("What the labels mean", icon=":material/label:")
    benign_col, malignant_col = st.columns(2, border=True)
    with benign_col:
        st.badge("Benign", color="green", icon=":material/check_circle:")
        st.write("Non-malignant tissue images from the source dataset.")
    with malignant_col:
        st.badge("Malignant", color="red", icon=":material/error:")
        st.write("Malignant tissue images from the source dataset.")
    st.caption("These labels describe dataset categories; they do not describe a person's health.")

    st.space("medium")
    with st.container(border=True):
        copy_col, art_col = st.columns([1.15, 0.85], vertical_alignment="center", gap="large")
        with copy_col:
            st.subheader("Early detection saves lives", icon=":material/favorite:")
            st.write(
                "Breast cancer outcomes improve with timely screening and clinical follow-up. "
                "Breast Scan AI is an educational exploration of how computer vision can support "
                "awareness around histopathology — it does not replace a clinician."
            )
            st.button(
                "Try the classifier",
                type="primary",
                icon=":material/play_arrow:",
                on_click=go_to,
                args=("Analyze",),
            )
        with art_col:
            st.image(
                BACKDROP_PATH,
                width="stretch",
                alt="Illustration of a pink awareness ribbon beside a mammogram on a monitor",
            )

    st.space("medium")
    st.warning(
        "Predictions are experimental model outputs. They must not be used for screening, diagnosis, "
        "treatment decisions, or reassurance about an individual's health.",
        title="Not a medical device",
        icon=":material/health_and_safety:",
    )

elif page == "Analyze":
    st.title("Analyze an image", icon=":material/upload_file:")
    st.caption("Educational use only — a model output, not a diagnosis.")

    model = get_model()
    if model is None:
        st.stop()

    st.space("small")
    upload_col, result_col = st.columns([1.05, 0.95], gap="large")

    with upload_col:
        with st.container(border=True, key="uploader"):
            st.subheader("Upload an image", icon=":material/add_photo_alternate:")
            st.caption("JPG, JPEG, or PNG. Use a single histopathology crop.")
            uploaded_file = st.file_uploader(
                "Histopathology image",
                type=["jpg", "jpeg", "png"],
                key=f"upload_{st.session_state.uploader_round}",
                label_visibility="collapsed",
            )

            image = None
            if uploaded_file is not None:
                try:
                    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                except Exception:
                    st.error(
                        "That file could not be opened as an image. Try a JPG or PNG file.",
                        icon=":material/broken_image:",
                    )

            if image is not None:
                st.image(
                    image,
                    width="stretch",
                    caption=(
                        f"{image.size[0]} × {image.size[1]} px · "
                        f"{image.format or 'unknown format'} · {uploaded_file.size / 1024:.0f} KB"
                    ),
                    alt="Preview of the uploaded histopathology image",
                )

                if st.button(
                    "Run analysis",
                    type="primary",
                    icon=":material/play_arrow:",
                    width="stretch",
                ):
                    with st.status("Analyzing image", expanded=False) as status:
                        batch = preprocess_image(image)
                        status.write("Converted to RGB and resized to 48 × 48 px")
                        scores = model.predict(batch, verbose=0)[0].astype(float)
                        status.update(label="Analysis complete", state="complete")

                    top_index = int(np.argmax(scores))
                    st.session_state.result = {
                        "signature": (uploaded_file.name, uploaded_file.size, st.session_state.uploader_round),
                        "verdict": CLASSES[top_index],
                        "score": float(scores[top_index]),
                        "benign": float(scores[0]),
                        "malignant": float(scores[1]),
                        "filename": uploaded_file.name,
                        "width": image.size[0],
                        "height": image.size[1],
                        "format": image.format or "Unknown",
                        "kilobytes": uploaded_file.size / 1024,
                        "input_shape": str(batch.shape),
                    }

                if st.session_state.result is not None:
                    st.button(
                        "Clear result",
                        icon=":material/delete:",
                        on_click=reset_analysis,
                        width="stretch",
                    )

            if uploaded_file is None:
                st.session_state.result = None
            elif (
                st.session_state.result is not None
                and st.session_state.result["signature"]
                != (uploaded_file.name, uploaded_file.size, st.session_state.uploader_round)
            ):
                st.session_state.result = None

    with result_col:
        st.subheader("Result", icon=":material/analytics:")
        render_result(st.session_state.result, image)

elif page == "About":
    st.title("About Breast Scan AI", icon=":material/menu_book:")
    st.caption("A collaborative AI capstone project from Army Public School, LBS Marg, Lucknow.")

    project_col, model_col = st.columns(2, gap="large")

    with project_col:
        with st.container(border=True, height="stretch"):
            st.subheader("The project", icon=":material/school:")
            st.write(
                "Breast Scan AI explores how deep learning can classify breast histopathology "
                "images as benign or malignant. It is an academic demonstration built on the "
                "public BreaKHis dataset, not a clinical product."
            )
            st.markdown(
                "- **Institution:** Army Public School, LBS Marg, Lucknow\n"
                "- **Academic batch:** AI Batch 2026–27\n"
                "- **Project guide:** Mrs. Seema Agarwal\n"
                "- **Project year:** 2026–2027"
            )

    with model_col:
        with st.container(border=True, height="stretch"):
            st.subheader("The model", icon=":material/neurology:")
            st.markdown(
                "| | |\n"
                "|---|---|\n"
                "| Architecture | Custom CNN |\n"
                "| Input | 48 × 48 px RGB |\n"
                "| Output | 2-class softmax (benign, malignant) |\n"
                "| Training data | BreaKHis — 7,909 images |\n"
                "| Framework | TensorFlow / Keras |\n"
                "| Reported accuracy | ≈92% (experimental) |"
            )

    st.space("small")
    with st.container(border=True):
        st.subheader("The team", icon=":material/groups:")
        st.markdown(
            "- **Raj Kumar** — Project lead, ML trainer and tester\n"
            "- **Daksha Naresh** — Data engineer\n"
            "- **Divyanshu Yadav** — UI designer\n"
            "- **Vishal Yadav** — Frontend developer\n"
            "- **Ansh Mishra** — Documentation lead"
        )

    st.space("small")
    with st.container(horizontal=True, gap="small"):
        st.link_button("Live demo", DEMO_URL, type="primary", icon=":material/rocket_launch:")
        st.link_button("GitHub", PROJECT_URL, icon=":material/code:")
        st.link_button("Email", f"mailto:{CONTACT_EMAIL}", icon=":material/mail:")

    st.space("small")
    st.warning(
        "Reported accuracy reflects the team's training results and is not independent clinical "
        "validation. The application is for education and research only.",
        title="Educational use only",
        icon=":material/health_and_safety:",
    )

    if os.path.exists(VIDEO_PATH):
        with st.expander("Watch the project walkthrough", icon=":material/play_circle:"):
            st.video(VIDEO_PATH, alt="Project walkthrough video for Breast Scan AI")

render_footer()
