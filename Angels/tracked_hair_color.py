import cv2
import mediapipe as mp
import torch
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

#  load face mesh model
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# load hair segmentation model
print("Loading hair segmentation model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
hair_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
hair_model.eval()
HAIR_LABEL = 2
PROCESS_SIZE = 256

# Anchor landmark indices (stable points: nose tip, eyes corners, chin, ears area)
ANCHOR_IDS = [1, 33, 263, 152, 234, 454]  # nose tip, left eye, right eye, chin, left cheek, right cheek

def get_anchors(face_landmarks, w, h):
    pts = []
    for idx in ANCHOR_IDS:
        lm = face_landmarks[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)

def get_hair_mask(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (PROCESS_SIZE, PROCESS_SIZE))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_small)
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = hair_model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1)[0].numpy()
    return (pred == HAIR_LABEL).astype(np.uint8)

def apply_color(frame, mask, color_bgr):
    if color_bgr is None:
        return frame
    color_layer = np.zeros_like(frame)
    color_layer[:] = color_bgr
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.merge([gray, gray, gray]).astype(np.float32) / 255.0
    tinted = (color_layer.astype(np.float32) * gray_3ch * 1.3).clip(0, 255)
    return (frame.astype(np.float32) * (1 - mask_3ch) + tinted * mask_3ch).astype(np.uint8)

base_mask = None
base_anchors = None
current_color = (0, 0, 200)
RESEG_INTERVAL = 30
frame_count = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Tracking started. Press 'q' to quit, 'r' to force re-segmentation.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    display_mask = None

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        current_anchors = get_anchors(face_landmarks, w, h)

        need_reseg = (base_mask is None) or (frame_count % RESEG_INTERVAL == 0)

        if need_reseg:
            base_mask = get_hair_mask(frame)
            base_anchors = current_anchors.copy()
            display_mask = base_mask
        else:
            transform_matrix, _ = cv2.estimateAffinePartial2D(base_anchors, current_anchors)
            if transform_matrix is not None:
                warped_mask = cv2.warpAffine(base_mask, transform_matrix, (w, h))
                display_mask = warped_mask
            else:
                display_mask = base_mask

        display = apply_color(frame, display_mask, current_color)
        cv2.putText(display, "Tracking OK" + (" (re-seg)" if need_reseg else ""), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        display = frame.copy()
        cv2.putText(display, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Angels - Tracked Hair Color', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        base_mask = None

cap.release()
cv2.destroyAllWindows()