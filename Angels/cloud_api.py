from flask import Flask
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


# load face mesh model 
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

# load hair segmentation model 
print("Loading hair segmentation model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
hair_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
hair_model.eval()
HAIR_LABEL = 2
PROCESS_SIZE = 256

ANCHOR_IDS = [1, 33, 263, 152, 234, 454]
RESEG_INTERVAL = 30

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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest] = 1
        mask = clean_mask

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, fw, fh) in faces:
        pad_x = int(fw * 0.05)
        pad_y_top = int(fh * 0.0)   # don't cut forehead (hair starts there)
        pad_y_bot = int(fh * 0.0)
        x1 = max(0, x + pad_x)
        y1 = max(0, y + int(fh * 0.25))  # start exclusion 25% down the face
        x2 = min(w, x + fw - pad_x)
        y2 = min(h, y + fh + pad_y_bot)
        mask[y1:y2, x1:x2] = 0

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
    
    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)
    
    # Convert hair region to grayscale to extract texture/luminance only
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Normalize luminance to 0.3-1.0 range so dark hair still shows color
    gray_normalized = 0.3 + gray * 0.7
    
    # Build the new hair color using luminance as a texture multiplier
    new_color = np.zeros_like(frame, dtype=np.float32)
    for c in range(3):
        new_color[:, :, c] = color_bgr[c] * gray_normalized
    
    # Apply softly smoothed mask for natural edges
    mask_smooth = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
    mask_3ch = cv2.merge([mask_smooth, mask_smooth, mask_smooth])
    
    # Replace hair color completely (high opacity blend — 0.85)
    OPACITY = 0.85
    result = result * (1 - mask_3ch * OPACITY) + new_color * (mask_3ch * OPACITY)
    
    return result.clip(0, 255).astype(np.uint8)


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}


@sock.route('/stream')
def stream(ws):
    """Live continuous color streaming with face-mesh tracked hair mask"""
    print("Client connected to stream")
    landmarker = vision.FaceLandmarker.create_from_options(face_options)

    base_mask = None
    base_anchors = None
    frame_count = 0
    current_color = None  # BGR tuple or None

    while True:
        try:
            raw = ws.receive()
            if raw is None:
                break
            msg = json.loads(raw)

            # Update selected color if sent
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

                need_reseg = (base_mask is None) or (frame_count % RESEG_INTERVAL == 0)

                if need_reseg:
                    base_mask = get_hair_mask(frame)
                    base_anchors = current_anchors.copy()
                    display_mask = base_mask
                else:
                    transform_matrix, _ = cv2.estimateAffinePartial2D(base_anchors, current_anchors)
                    if transform_matrix is not None:
                        display_mask = cv2.warpAffine(base_mask, transform_matrix, (w, h))
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
    from flask import request, jsonify
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