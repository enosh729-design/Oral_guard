import streamlit as st
import hmac
import io
import time
import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os
from pathlib import Path
from datetime import datetime
from huggingface_hub import hf_hub_download

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="OralGuard — Dental AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Password gate ─────────────────────────────────────────────
def check_password():
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["APP_PASSWORD"]
        ):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("## 🦷 OralGuard — Dental Pathology Detection")
    st.markdown("*Research demonstration tool. Not for clinical use.*")
    st.text_input(
        "Enter access password",
        type="password",
        on_change=password_entered,
        key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Incorrect password. Contact Dr. Enosh A. Paulson for access.")
    st.markdown("---")
    st.caption(
        "⚠️ Do not upload X-rays containing patient-identifying information. "
        "All uploads are processed in memory and immediately discarded. "
        "For research and educational use only."
    )
    return False

if not check_password():
    st.stop()

# ── Rate limiting ─────────────────────────────────────────────
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
    st.session_state.first_request_time = time.time()

if time.time() - st.session_state.first_request_time > 3600:
    st.session_state.request_count = 0
    st.session_state.first_request_time = time.time()

if st.session_state.request_count >= 20:
    st.error("Rate limit reached — maximum 20 analyses per hour per session.")
    st.stop()

# ── Load models from HuggingFace Hub ─────────────────────────
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    from src.classifier.model import get_model

    with st.spinner("Loading OralGuard models from HuggingFace Hub..."):
        yolo_path = hf_hub_download(
            repo_id="Enosh729/oralguard",
            filename="oralguard_det_finetuned.pt"
        )
        clf_path = hf_hub_download(
            repo_id="Enosh729/oralguard",
            filename="classifier_finetuned.pt"
        )

    yolo = YOLO(yolo_path)
    clf = get_model(pretrained=False, weights_path=clf_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = clf.to(device)
    clf.eval()
    return yolo, clf, device

sys.path.insert(0, str(Path(__file__).parent))
yolo_model, clf_model, DEVICE = load_models()

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES  = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
CLASS_LABELS = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]
THRESHOLDS = {
    "caries": 0.50,
    "deep_caries": 0.40,
    "periapical_lesion": 0.30,
    "impacted_tooth": 0.35
}
UNCERTAINTY_THRESHOLD = 2.0

# ── Pathology colors (BGR for OpenCV) ─────────────────────────
PATHOLOGY_COLORS = {
    "caries":            (255, 105, 180),
    "deep_caries":       (0, 165, 255),
    "periapical_lesion": (147, 112, 219),
    "impacted_tooth":    (60, 60, 255),
    "none":              (128, 128, 128),
}

# ── FDI tooth clinical names ──────────────────────────────────
FDI_TOOTH_NAMES = {
    11: "UR Central Incisor",  12: "UR Lateral Incisor",
    13: "UR Canine",           14: "UR 1st Premolar",
    15: "UR 2nd Premolar",     16: "UR 1st Molar",
    17: "UR 2nd Molar",        18: "UR 3rd Molar",
    21: "UL Central Incisor",  22: "UL Lateral Incisor",
    23: "UL Canine",           24: "UL 1st Premolar",
    25: "UL 2nd Premolar",     26: "UL 1st Molar",
    27: "UL 2nd Molar",        28: "UL 3rd Molar",
    31: "LL Central Incisor",  32: "LL Lateral Incisor",
    33: "LL Canine",           34: "LL 1st Premolar",
    35: "LL 2nd Premolar",     36: "LL 1st Molar",
    37: "LL 2nd Molar",        38: "LL 3rd Molar",
    41: "LR Central Incisor",  42: "LR Lateral Incisor",
    43: "LR Canine",           44: "LR 1st Premolar",
    45: "LR 2nd Premolar",     46: "LR 1st Molar",
    47: "LR 2nd Molar",        48: "LR 3rd Molar",
}

# ── Inference pipeline ────────────────────────────────────────
def run_pipeline(image_bytes):
    from src.detector.fdi_mapper import map_to_fdi
    from src.classifier.uncertainty import mc_uncertainty

    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w      = img_rgb.shape[:2]

    results = yolo_model.predict(
        source=img_rgb, imgsz=1024, conf=0.25, iou=0.45,
        agnostic_nms=True,
        device=0 if DEVICE == "cuda" else "cpu",
        verbose=False
    )

    boxes     = results[0].boxes.xyxy.cpu().numpy()
    annotated = img_rgb.copy()
    seen_fdi  = {}

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
        fdi = map_to_fdi(cx, cy, x2 - x1, y2 - y1, w, h)

        patch = img_rgb[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        patch_t = torch.tensor(
            cv2.resize(patch, (128, 128)).transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE) / 255.0

        mean_pred, entropy = mc_uncertainty(clf_model, patch_t, T=30)
        probs     = mean_pred[0].cpu().numpy()
        uncertain = bool(entropy.mean().item() > UNCERTAINTY_THRESHOLD)

        tooth_findings = [
            CLASS_NAMES[i] for i, p in enumerate(probs)
            if p >= THRESHOLDS[CLASS_NAMES[i]]
        ]
        confidence = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        entry = {
            "fdi": fdi, "findings": tooth_findings,
            "confidence": confidence, "uncertain": uncertain,
            "box": (x1, y1, x2, y2),
            "entropy": float(entropy.mean().item())
        }

        if fdi not in seen_fdi:
            seen_fdi[fdi] = entry
        elif confidence["caries"] > seen_fdi[fdi]["confidence"]["caries"]:
            seen_fdi[fdi] = entry

    findings = list(seen_fdi.values())
    findings_sorted = sorted(findings, key=lambda x: x["fdi"])

    # ── Annotate: reference dental-AI style ──────────────────────
    # Only teeth WITH findings get a very subtle color tint + thin
    # outline. Teeth with no pathology are left completely clean.
    # NO text, NO numbers, NO labels on the X-ray at all.
    overlay = annotated.copy()

    for idx, f in enumerate(findings_sorted, start=1):
        f["marker_num"] = idx
        x1, y1, x2, y2 = f["box"]

        # Only color teeth that actually have findings
        if not f["findings"]:
            continue

        primary = f["findings"][0]
        color = PATHOLOGY_COLORS.get(primary, PATHOLOGY_COLORS["none"])

        # Subtle filled rectangle on the overlay copy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend at 15% — just a hint of color, X-ray stays readable
    annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)

    # Thin outlines on top (only for teeth with findings)
    for f in findings_sorted:
        if not f["findings"]:
            continue
        x1, y1, x2, y2 = f["box"]
        primary = f["findings"][0]
        color = PATHOLOGY_COLORS.get(primary, PATHOLOGY_COLORS["none"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # ── Compact legend bar ────────────────────────────────────
    legend_h = 32
    h_img, w_img = annotated.shape[:2]
    legend = np.zeros((legend_h, w_img, 3), dtype=np.uint8)
    legend[:] = (18, 18, 18)

    legend_items = [
        ("Caries", PATHOLOGY_COLORS["caries"]),
        ("Deep Caries", PATHOLOGY_COLORS["deep_caries"]),
        ("Periapical Lesion", PATHOLOGY_COLORS["periapical_lesion"]),
        ("Impacted", PATHOLOGY_COLORS["impacted_tooth"]),
    ]
    total_w = sum(14 + len(l) * 7 + 16 for l, _ in legend_items)
    x_off = max(10, (w_img - total_w) // 2)    # center the legend
    for label, clr in legend_items:
        cv2.circle(legend, (x_off + 5, legend_h // 2), 4, clr, -1)
        cv2.putText(
            legend, label, (x_off + 14, legend_h // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA
        )
        x_off += 14 + len(label) * 7 + 16

    annotated = np.vstack([annotated, legend])

    return findings_sorted, annotated

# ── PDF Report Generator ──────────────────────────────────────
def generate_pdf_report(findings, annotated_img, filename, elapsed):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="OralGuard Dental Analysis Report"
    )

    W, H      = A4
    PAGE_W    = W - 4*cm
    styles    = getSampleStyleSheet()
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── Custom styles ─────────────────────────────
    NAVY   = colors.HexColor("#0a1628")
    BLUE   = colors.HexColor("#1a4a8a")
    CYAN   = colors.HexColor("#2196F3")
    GREEN  = colors.HexColor("#2e7d32")
    AMBER  = colors.HexColor("#f57c00")
    RED    = colors.HexColor("#c62828")
    LGRAY  = colors.HexColor("#f5f5f5")
    MGRAY  = colors.HexColor("#e0e0e0")
    DGRAY  = colors.HexColor("#424242")

    s_title = ParagraphStyle(
        "ReportTitle", fontSize=22, textColor=colors.white,
        fontName="Helvetica-Bold", alignment=TA_CENTER, leading=28
    )
    s_sub = ParagraphStyle(
        "ReportSub", fontSize=10, textColor=colors.HexColor("#bbdefb"),
        fontName="Helvetica", alignment=TA_CENTER, leading=14
    )
    s_h2 = ParagraphStyle(
        "H2", fontSize=13, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6, leading=18
    )
    s_h3 = ParagraphStyle(
        "H3", fontSize=11, textColor=BLUE,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4, leading=15
    )
    s_body = ParagraphStyle(
        "Body", fontSize=9.5, textColor=DGRAY,
        fontName="Helvetica", leading=14, alignment=TA_JUSTIFY
    )
    s_small = ParagraphStyle(
        "Small", fontSize=8, textColor=colors.HexColor("#757575"),
        fontName="Helvetica", leading=11
    )
    s_bold = ParagraphStyle(
        "Bold", fontSize=9.5, textColor=DGRAY,
        fontName="Helvetica-Bold", leading=14
    )
    s_disclaimer = ParagraphStyle(
        "Disclaimer", fontSize=8, textColor=colors.HexColor("#b71c1c"),
        fontName="Helvetica-Oblique", leading=11, alignment=TA_CENTER
    )

    story = []

    # ── Header banner ─────────────────────────────
    header_data = [[
        Paragraph("🦷  OralGuard Dental Analysis Report", s_title),
    ]]
    header_table = Table(header_data, colWidths=[PAGE_W])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), NAVY),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), [8,8,8,8]),
        ("TOPPADDING",  (0,0), (-1,-1), 18),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 16),
        ("RIGHTPADDING",(0,0), (-1,-1), 16),
    ]))
    story.append(header_table)

    sub_data = [[
        Paragraph(
            "Uncertainty-Aware Multi-Task Pathology Detection  |  "
            "YOLOv8m + ResNet50 + Monte Carlo Dropout (T=30)",
            s_sub
        )
    ]]
    sub_table = Table(sub_data, colWidths=[PAGE_W])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), BLUE),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 16),
        ("RIGHTPADDING",(0,0), (-1,-1), 16),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 14))

    # ── Report metadata ───────────────────────────
    n_teeth    = len(findings)
    n_findings = sum(1 for f in findings if f["findings"])
    n_uncertain= sum(1 for f in findings if f["uncertain"])

    meta_rows = [
        ["Report Generated", timestamp,  "File Analysed",   filename],
        ["Teeth Detected",   str(n_teeth), "Analysis Time", f"{elapsed:.1f} seconds"],
        ["Teeth with Findings", str(n_findings), "Uncertain Flags", str(n_uncertain)],
        ["Model (Detector)", "YOLOv8m  mAP@50: 0.548",
         "Model (Classifier)", "ResNet50  F1: 0.564"],
    ]

    meta_col_styles = [PAGE_W*0.22, PAGE_W*0.28, PAGE_W*0.22, PAGE_W*0.28]
    meta_table = Table(meta_rows, colWidths=meta_col_styles)
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), LGRAY),
        ("BACKGROUND",  (2,0), (2,-1), LGRAY),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTNAME",    (1,0), (1,-1), "Helvetica"),
        ("FONTNAME",    (3,0), (3,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("TEXTCOLOR",   (0,0), (-1,-1), DGRAY),
        ("TEXTCOLOR",   (0,0), (0,-1), BLUE),
        ("TEXTCOLOR",   (2,0), (2,-1), BLUE),
        ("GRID",        (0,0), (-1,-1), 0.4, MGRAY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, LGRAY]),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── Annotated image ───────────────────────────
    story.append(Paragraph("Annotated Radiograph", s_h2))
    story.append(HRFlowable(width=PAGE_W, thickness=1, color=CYAN, spaceAfter=8))

    pil_ann = Image.fromarray(annotated_img)
    ratio   = pil_ann.height / pil_ann.width
    img_w   = PAGE_W
    img_h   = min(img_w * ratio, 11*cm)
    img_buf = io.BytesIO()
    pil_ann.save(img_buf, format="JPEG", quality=88)
    img_buf.seek(0)
    rl_img = RLImage(img_buf, width=img_w, height=img_h)
    story.append(rl_img)
    story.append(Paragraph(
        "Green boxes = Confident prediction  |  Red boxes = High uncertainty (recommend expert review)",
        s_small
    ))
    story.append(Spacer(1, 14))

    # ── Summary statistics ────────────────────────
    story.append(Paragraph("Summary Statistics", s_h2))
    story.append(HRFlowable(width=PAGE_W, thickness=1, color=CYAN, spaceAfter=8))

    class_counts = {c: 0 for c in CLASS_NAMES}
    for f in findings:
        for c in f["findings"]:
            class_counts[c] += 1

    stat_rows = [
        ["Pathology Class", "Detected Count", "Threshold Used", "Status"]
    ] + [
        [
            lbl,
            str(class_counts[cls]),
            f"{THRESHOLDS[cls]:.0%}",
            "Detected" if class_counts[cls] > 0 else "Not detected"
        ]
        for cls, lbl in zip(CLASS_NAMES, CLASS_LABELS)
    ]

    stat_table = Table(stat_rows, colWidths=[PAGE_W*0.35, PAGE_W*0.2, PAGE_W*0.2, PAGE_W*0.25])
    stat_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("GRID",         (0,0), (-1,-1), 0.4, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LGRAY]),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 14))

    # ── Per-tooth findings ────────────────────────
    story.append(Paragraph("Per-Tooth Detailed Findings", s_h2))
    story.append(HRFlowable(width=PAGE_W, thickness=1, color=CYAN, spaceAfter=8))

    sorted_findings = sorted(findings, key=lambda x: x["fdi"])

    teeth_rows = [[
        "FDI #", "Findings", "Caries", "Deep Caries",
        "Periapical\nLesion", "Impacted\nTooth", "Entropy", "Status"
    ]]
    for f in sorted_findings:
        findings_str = ", ".join(f["findings"]).replace("_", " ").title() \
                       if f["findings"] else "None"
        status = "UNCERTAIN" if f["uncertain"] else "Confident"
        c = f["confidence"]
        teeth_rows.append([
            str(f["fdi"]),
            findings_str,
            f"{c['caries']:.1%}",
            f"{c['deep_caries']:.1%}",
            f"{c['periapical_lesion']:.1%}",
            f"{c['impacted_tooth']:.1%}",
            f"{f['entropy']:.2f}",
            status
        ])

    col_ws = [
        PAGE_W*0.07, PAGE_W*0.20, PAGE_W*0.09, PAGE_W*0.10,
        PAGE_W*0.10, PAGE_W*0.09, PAGE_W*0.09, PAGE_W*0.10
    ]
    teeth_table = Table(teeth_rows, colWidths=col_ws, repeatRows=1)

    # Build row-by-row colours
    teeth_styles = [
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.3, MGRAY),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ALIGN",         (1,0), (1,-1),  "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [colors.white, LGRAY]),
        ("WORDWRAP",      (0,0), (-1,-1), True),
    ]
    for i, f in enumerate(sorted_findings, start=1):
        if f["uncertain"]:
            teeth_styles.append(("BACKGROUND", (7,i), (7,i), colors.HexColor("#fff3e0")))
            teeth_styles.append(("TEXTCOLOR",  (7,i), (7,i), AMBER))
            teeth_styles.append(("FONTNAME",   (7,i), (7,i), "Helvetica-Bold"))
        if f["findings"]:
            teeth_styles.append(("TEXTCOLOR",  (1,i), (1,i), RED))
            teeth_styles.append(("FONTNAME",   (1,i), (1,i), "Helvetica-Bold"))

    teeth_table.setStyle(TableStyle(teeth_styles))
    story.append(teeth_table)
    story.append(Spacer(1, 14))

    # ── Clinical notes ────────────────────────────
    story.append(Paragraph("Clinical Interpretation Notes", s_h2))
    story.append(HRFlowable(width=PAGE_W, thickness=1, color=CYAN, spaceAfter=8))

    notes = [
        ("<b>Detection Thresholds:</b> Class-specific confidence thresholds were applied — "
         "Caries (50%), Deep Caries (40%), Periapical Lesion (30%), Impacted Tooth (35%). "
         "Lower thresholds for minority classes improve sensitivity for rare pathologies."),
        ("<b>Uncertainty Quantification:</b> Monte Carlo Dropout (T=30 stochastic passes) "
         "was used to estimate predictive entropy. Teeth with entropy > 2.0 nats are flagged "
         "UNCERTAIN and should be reviewed by a senior clinician. Max possible entropy for "
         "this 4-class model is 4×ln(2) ≈ 2.77 nats."),
        ("<b>FDI Notation:</b> Tooth numbering follows the ISO 3950 / FDI World Dental "
         "Federation two-digit notation. Quadrants: 1=UR, 2=UL, 3=LL, 4=LR."),
        ("<b>Recommendation:</b> All findings flagged by OralGuard should be confirmed "
         "with clinical examination and additional radiographic views as appropriate. "
         "This report does not constitute a clinical diagnosis."),
    ]
    for note in notes:
        story.append(Paragraph(note, s_body))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 10))

    # ── Disclaimer ────────────────────────────────
    disc_data = [[
        Paragraph(
            "⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS  |  "
            "OralGuard is an AI research tool trained on the DENTEX 2023 dataset. "
            "It has not been validated for clinical use. Always consult a qualified "
            "dental professional. Generated by OralGuard v1.0 — Dr. Enosh A. Paulson, "
            "BDS (RGUHS) | PGDMI Candidate, IIHMR Bangalore.",
            s_disclaimer
        )
    ]]
    disc_table = Table(disc_data, colWidths=[PAGE_W])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#ffebee")),
        ("BOX",          (0,0), (-1,-1), 0.8, RED),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(disc_table)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── UI ────────────────────────────────────────────────────────
st.markdown("# 🦷 OralGuard")
st.markdown(
    "**Uncertainty-Aware Multi-Task Dental Pathology Detection** | "
    "YOLOv8 + ResNet50 + Monte Carlo Dropout"
)
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Panoramic OPG")
    st.caption(
        "⚠️ Research tool only. Do not upload patient-identifiable images. "
        "All processing is in-memory — nothing is stored."
    )

    uploaded = st.file_uploader(
        "Upload OPG (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("File too large. Maximum 10MB.")
            st.stop()

        try:
            image_bytes = uploaded.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            image_bytes = uploaded.getvalue()
        except Exception:
            st.error("Invalid image file. Please upload a valid JPG or PNG.")
            st.stop()

        st.image(image_bytes, caption="Uploaded OPG", use_container_width=True)

with col2:
    if uploaded:
        st.markdown("### Analysis Results")

        with st.spinner("Running OralGuard pipeline..."):
            start = time.time()
            findings, annotated_img = run_pipeline(image_bytes)
            elapsed = time.time() - start
            st.session_state.request_count += 1

        st.image(
            annotated_img,
            caption=f"Detected teeth with findings ({elapsed:.1f}s)",
            use_container_width=True
        )

        st.markdown(f"**{len(findings)} teeth detected** in {elapsed:.1f}s")
        st.markdown("---")

        if not findings:
            st.info("No teeth detected. Try a clearer panoramic OPG.")
        else:
            # ── Findings HTML table ───────────────────
            BADGE_COLORS_HEX = {
                "caries":            "#FF69B4",
                "deep_caries":       "#FFA500",
                "periapical_lesion": "#9370DB",
                "impacted_tooth":    "#FF3C3C",
            }

            def _badge(cls_name):
                """Return an HTML badge span for a pathology class."""
                color = BADGE_COLORS_HEX.get(cls_name, "#888")
                label = cls_name.replace("_", " ").title()
                return (
                    f'<span style="background:{color};color:#fff;'
                    f'padding:2px 8px;border-radius:10px;font-size:0.82em;'
                    f'font-weight:600;margin-right:4px;">{label}</span>'
                )

            rows_html = ""
            for f in findings:                    # already sorted by fdi
                num = f.get("marker_num", "?")
                fdi = f["fdi"]
                tooth_name = FDI_TOOTH_NAMES.get(fdi, f"Tooth {fdi}")

                if f["findings"]:
                    badges = " ".join(_badge(c) for c in f["findings"])
                else:
                    badges = (
                        '<span style="color:#888;font-style:italic;">'
                        'Within normal limits</span>'
                    )

                # Max confidence among detected findings
                if f["findings"]:
                    max_conf = max(f["confidence"][c] for c in f["findings"])
                    conf_str = f"{max_conf:.0%}"
                else:
                    conf_str = "—"

                if f["uncertain"]:
                    status = '<span style="color:#f5a623;font-weight:700;">⚠️ Review</span>'
                else:
                    status = '<span style="color:#27ae60;font-weight:700;">✓ Check</span>'

                rows_html += (
                    f"<tr>"
                    f'<td style="text-align:center;font-weight:700;">{num}</td>'
                    f'<td style="text-align:center;">{fdi}</td>'
                    f"<td>{tooth_name}</td>"
                    f"<td>{badges}</td>"
                    f'<td style="text-align:center;">{conf_str}</td>'
                    f'<td style="text-align:center;">{status}</td>'
                    f"</tr>"
                )

            table_html = (
                '<div style="overflow-x:auto;">'
                '<table style="width:100%;border-collapse:collapse;font-size:0.92em;">'
                "<thead><tr style='background:#0a1628;color:#fff;'>"
                '<th style="padding:8px 10px;">#</th>'
                '<th style="padding:8px 10px;">FDI</th>'
                '<th style="padding:8px 10px;text-align:left;">Tooth Name</th>'
                '<th style="padding:8px 10px;text-align:left;">Findings</th>'
                '<th style="padding:8px 10px;">Confidence</th>'
                '<th style="padding:8px 10px;">Status</th>'
                "</tr></thead>"
                f"<tbody>{rows_html}</tbody>"
                "</table></div>"
            )

            # Alternating row colours via inline CSS
            table_css = (
                "<style>"
                "table tbody tr:nth-child(even){background:#f5f7fa;}"
                "table tbody tr:nth-child(odd){background:#fff;}"
                "table tbody tr:hover{background:#e8f0fe;}"
                "table td,table th{padding:8px 10px;border-bottom:1px solid #e0e0e0;}"
                "</style>"
            )
            st.markdown(table_css + table_html, unsafe_allow_html=True)

            # ── About this analysis (collapsed) ──────
            with st.expander("ℹ️ About this analysis", expanded=False):
                st.markdown(
                    "**Model Architecture:** YOLOv8m (tooth detection, mAP@50 0.548) "
                    "+ ResNet50 (multi-label classification, F1 0.564)"
                )
                st.markdown(
                    "**Uncertainty Quantification:** Monte Carlo Dropout "
                    "(T=30 stochastic forward passes). Predictive entropy > 2.0 nats "
                    "flags a tooth as *uncertain*. Max possible entropy for this "
                    "4-class model is 4×ln(2) ≈ 2.77 nats."
                )
                st.markdown("**Detection Thresholds:**")
                thresh_md = " | ".join(
                    f"{lbl}: {THRESHOLDS[c]:.0%}"
                    for c, lbl in zip(CLASS_NAMES, CLASS_LABELS)
                )
                st.markdown(f"  {thresh_md}")
                st.markdown("**Per-tooth entropy scores:**")
                for f in findings:
                    st.caption(
                        f"Tooth {f['fdi']} ({FDI_TOOTH_NAMES.get(f['fdi'], '?')}): "
                        f"entropy = {f['entropy']:.3f} nats"
                    )

        # ── PDF Download ──────────────────────────
        st.markdown("---")
        st.markdown("#### 📄 Download Full Report")

        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_pdf_report(
                    findings, annotated_img,
                    uploaded.name, elapsed
                )
                report_name = (
                    f"OralGuard_Report_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                st.download_button(
                    label="⬇️  Download PDF Report",
                    data=pdf_bytes,
                    file_name=report_name,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                st.caption(
                    f"Report includes annotated radiograph, per-tooth confidence "
                    f"scores, entropy values, and clinical notes."
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        st.markdown("---")
        st.caption(
            "**Disclaimer:** OralGuard is a research demonstration. "
            "Not validated for clinical use. Always consult a qualified "
            "dental professional for diagnosis."
        )
