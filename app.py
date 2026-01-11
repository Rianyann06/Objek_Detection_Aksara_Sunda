# -*- coding: utf-8 -*-

# =========================
# ENV FIX (WAJIB DI ATAS)
# =========================
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

# Import ultralytics SETELAH env fix
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi Aksara Sunda (YOLOv11)",
    layout="centered"
)

MODEL_PATH = "best.pt"
CLASS_NAMES_PATH = "class_names.txt"

CONF_THRES = 0.35
IOU_THRES = 0.2
NMS_IOU = 0.20

# =========================
# LOAD CLASS NAMES
# =========================
@st.cache_resource(show_spinner=False)
def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# =========================
# LOAD MODEL (SAFE)
# =========================
@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO(MODEL_PATH)

# =========================
# UTILS
# =========================
def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def ultra_aggressive_nms(boxes, scores, classes, iou_thresh=0.20):
    if not boxes:
        return [], [], []

    idxs = np.argsort(scores)[::-1]
    keep = []
    removed = set()

    for i in idxs:
        if i in removed:
            continue
        keep.append(i)
        for j in idxs:
            if j <= i or j in removed:
                continue
            if iou_xyxy(boxes[i], boxes[j]) > iou_thresh:
                removed.add(j)

    return (
        [boxes[i] for i in keep],
        [scores[i] for i in keep],
        [classes[i] for i in keep],
    )

def sort_reading_order(boxes, scores, classes):
    centers_x = [(b[0] + b[2]) / 2 for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: centers_x[i])
    return (
        [boxes[i] for i in order],
        [scores[i] for i in order],
        [classes[i] for i in order],
    )

def add_smart_spacing(boxes, classes, class_names):
    if not boxes:
        return ""

    widths = [b[2] - b[0] for b in boxes]
    median_width = np.median(widths)

    result = []
    for i, cls in enumerate(classes):
        result.append(class_names[cls])
        if i < len(boxes) - 1:
            gap = boxes[i + 1][0] - boxes[i][2]
            if gap > median_width * 0.35:
                result.append(" ")

    return "".join(result)

# =========================
# DRAW BOXES
# =========================
def draw_boxes(image, boxes, scores, classes, class_names):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for idx, ((x1, y1, x2, y2), sc, cl) in enumerate(zip(boxes, scores, classes)):
        label = f"{idx+1}:{class_names[cl]} {sc:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1 + 3, y1 - 15), label, fill="green")

    return img

# =========================
# MAIN APP
# =========================
st.title("🔠 Deteksi Aksara Sunda Tulisan Tangan (YOLOv11)")

if not os.path.exists(MODEL_PATH):
    st.error("❌ File model 'best.pt' tidak ditemukan.")
    st.stop()

if not os.path.exists(CLASS_NAMES_PATH):
    st.error("❌ File 'class_names.txt' tidak ditemukan.")
    st.stop()

class_names = load_class_names(CLASS_NAMES_PATH)
model = load_model()

uploaded_file = st.file_uploader(
    "📤 Upload gambar aksara Sunda",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Gambar Input", use_container_width=True)

    if st.button("🔍 Deteksi Aksara"):
        with st.spinner("⏳ Mendeteksi..."):
            results = model.predict(
                image,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=640,
                verbose=False
            )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            st.warning("Tidak ada aksara terdeteksi.")
            st.stop()

        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        scores = r.boxes.conf.cpu().numpy().tolist()
        classes = r.boxes.cls.cpu().numpy().astype(int).tolist()

        boxes, scores, classes = ultra_aggressive_nms(
            boxes, scores, classes, iou_thresh=NMS_IOU
        )

        boxes, scores, classes = sort_reading_order(
            boxes, scores, classes
        )

        hasil_teks = add_smart_spacing(
            boxes, classes, class_names
        )

        result_img = draw_boxes(
            image, boxes, scores, classes, class_names
        )

        st.image(result_img, caption="✅ Hasil Deteksi", use_container_width=True)
        st.subheader("📄 Hasil Pembacaan")
        st.success(hasil_teks)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("YOLOv11 • Ultra-Aggressive NMS • Reading Order • Smart Spacing")
