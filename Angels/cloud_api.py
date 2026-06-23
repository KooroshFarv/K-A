from flask import Flask, request, jsonify
from flask_sock import Sock
import cv2
import numpy as np
import base64
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import time
import os
import json
import urllib.request

app = Flask(__name__)
sock = Sock(app)

API_KEY = os.environ.get("LIGHTX_API_KEY")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face mesh model
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load hair segmentation model
print("Loading hair segmentation model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
hair_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
hair_model.eval()
HAIR_LABEL = 2
PROCESS_SIZE = 256

ANCHOR_IDS = [1, 33, 263, 152, 234, 454]

def decode_frame(frame_b64):
    frame_bytes = base64.b64decode(frame_b64)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

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
    mask = (pred == HAIR_LABEL).astype(np.uint8)

    # Remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest] = 1
        mask = clean_mask

    # Fill holes inside hair region
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    # Exclude face region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, fw, fh) in faces:
        pad_x = int(fw * 0.05)
        x1 = max(0, x + pad_x)
        y1 = max(0, y + int(fh * 0.15))
        x2 = min(w, x + fw - pad_x)
        y2 = min(h, y + fh + int(fh * 0.1))
        mask[y1:y2, x1:x2] = 0

    # Smooth edges
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
    mask = (mask_blur > 0.5).astype(np.uint8)

    return mask

def get_anchors(face_landmarks, w, h):
    pts = []
    for idx in ANCHOR_IDS:
        lm = face_landmarks[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)

def apply_color(frame, mask, color_bgr):
    if color_bgr is None:
        return frame

    # Convert frame and target color to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    color_pixel = np.uint8([[list(color_bgr)]])
    color_hsv = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)

    target_hue = color_hsv[0]
    target_sat = color_hsv[1]

    output_hsv = frame_hsv.copy()

    # Smoothed mask for natural edges
    mask_smooth = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)

    # Replace hue completely where mask is active
    output_hsv[:, :, 0] = frame_hsv[:, :, 0] * (1 - mask_smooth) + target_hue * mask_smooth

    # Boost saturation toward target color
    value_map = frame_hsv[:, :, 2] / 255.0
    sat_boost = target_sat * (0.5 + 0.5 * value_map)
    output_hsv[:, :, 1] = np.clip(
        frame_hsv[:, :, 1] * (1 - mask_smooth) + sat_boost * mask_smooth,
        0, 255
    )

    # Keep original Value (brightness) to preserve hair texture
    output_bgr = cv2.cvtColor(output_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Final blend
    mask_3ch = cv2.merge([mask_smooth, mask_smooth, mask_smooth])
    result = (frame.astype(np.float32) * (1 - mask_3ch * 0.9) +
              output_bgr.astype(np.float32) * (mask_3ch * 0.9))

    return result.clip(0, 255).astype(np.uint8)


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}


@sock.route('/stream')
def stream(ws):
    print("Client connected to stream")
    landmarker = vision.FaceLandmarker.create_from_options(face_options)

    base_mask = None
    base_anchors = None
    frame_count = 0
    current_color = None
    smooth_matrix = None
    SMOOTH_FACTOR = 0.85

    while True:
        try:
            raw = ws.receive()
            if raw is None:
                break
            msg = json.loads(raw)

            if "color_bgr" in msg:
                current_color = tuple(msg["color_bgr"]) if msg["color_bgr"] else None

            if "frame" not in msg:
                continue

            frame = decode_frame(msg["frame"])
            h, w = frame.shape[:2]
            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                current_anchors = get_anchors(face_landmarks, w, h)

                need_reseg = (base_mask is None) or (frame_count % 60 == 0)

                if need_reseg:
                    base_mask = get_hair_mask(frame)
                    base_anchors = current_anchors.copy()
                    smooth_matrix = None
                    display_mask = base_mask
                else:
                    transform_matrix, _ = cv2.estimateAffinePartial2D(base_anchors, current_anchors)
                    if transform_matrix is not None:
                        if smooth_matrix is None:
                            smooth_matrix = transform_matrix
                        else:
                            smooth_matrix = (SMOOTH_FACTOR * smooth_matrix +
                                           (1 - SMOOTH_FACTOR) * transform_matrix)
                        display_mask = cv2.warpAffine(base_mask, smooth_matrix, (w, h))
                    else:
                        display_mask = base_mask

                output = apply_color(frame, display_mask, current_color)
            else:
                output = frame

            ws.send(json.dumps({"frame": encode_frame(output)}))

        except Exception as e:
            print(f"Stream error: {e}")
            break

    print("Client disconnected")


@app.route('/hairstyle', methods=['POST'])
def hairstyle():
    data = request.json
    frame = decode_frame(data['frame'])
    hairstyle_name = data['hairstyle']
    cv2.imwrite("temp_cloud_photo.jpg", frame)
    with open("temp_cloud_photo.jpg", "rb") as f:
        upload_response = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f})
    upload_data = upload_response.json()
    image_url = upload_data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    payload = {"imageUrl": image_url, "textPrompt": hairstyle_name}
    response = requests.post("https://api.lightxeditor.com/external/api/v1/hairstyle", headers=headers, json=payload)
    if response.status_code != 200:
        return jsonify({"error": f"API error: {response.status_code}"}), 500
    order_id = response.json()["body"]["orderId"]
    for i in range(20):
        time.sleep(5)
        poll = requests.post("https://api.lightxeditor.com/external/api/v1/order-status", headers=headers, json={"orderId": order_id})
        poll_data = poll.json()
        if poll_data.get("body", {}).get("status") == "active":
            result_url = poll_data["body"].get("output")
            if result_url:
                img_response = requests.get(result_url)
                nparr = np.frombuffer(img_response.content, np.uint8)
                result_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return jsonify({"frame": encode_frame(result_frame)})
    return jsonify({"error": "Timed out"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)