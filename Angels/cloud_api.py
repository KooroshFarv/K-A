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
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# face mesh model
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

# hair segmentation model
print("Loading hair segmentation model...")
from transformers import SegformerForSemanticSegmentation
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
hair_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
hair_model.eval()
HAIR_LABEL = 13  # label 13 = hair in jonathandinu/face-parsing (CelebAMask-HQ labels)
NECK_LABEL = 17
SKIN_LABEL = 1
PROCESS_SIZE = 512

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
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].numpy()

    # Hair mask — label 13
    mask = (pred == HAIR_LABEL).astype(np.uint8)

    # Remove noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest] = 1
        mask = clean_mask

    # Fill holes
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    # Use model's own predictions to exclude face regions
    # This is much more accurate than Haar cascade or YCrCb skin detection
    # because the same model that finds hair also finds skin/neck/cloth
    skin_region = (pred == SKIN_LABEL).astype(np.uint8)   # face skin
    neck_region = (pred == NECK_LABEL).astype(np.uint8)   # neck
    cloth_region = (pred == 18).astype(np.uint8)           # clothing

    # Dilate exclusion zones slightly to create a buffer at boundaries
    excl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_excl = cv2.dilate(skin_region, excl_kernel, iterations=1)
    neck_excl = cv2.dilate(neck_region, excl_kernel, iterations=2)
    cloth_excl = cv2.dilate(cloth_region, excl_kernel, iterations=1)

    # Remove those regions from hair mask
    mask[skin_excl > 0] = 0
    mask[neck_excl > 0] = 0
    mask[cloth_excl > 0] = 0

    # Detect hair type for adaptive processing
    hair_pixels = frame[mask == 1]
    if len(hair_pixels) > 100:
        hair_hsv = cv2.cvtColor(
            hair_pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)
        avg_val = np.median(hair_hsv[:, 2])
        avg_sat = np.median(hair_hsv[:, 1])
        avg_hue = np.median(hair_hsv[:, 0])
        if avg_val < 80:
            hair_type = "dark"
        elif (avg_hue < 25 or avg_hue > 155) and avg_sat > 60:
            hair_type = "warm"
        elif avg_val > 160 and avg_sat < 80:
            hair_type = "light"
        else:
            hair_type = "medium"
    else:
        hair_type = "medium"

    # Soft feathered edges
    mask_float = mask.astype(np.float32)
    blur_size = 21 if hair_type == "warm" else 31
    mask_blur = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    return mask_blur

def get_anchors(face_landmarks, w, h):
    pts = []
    for idx in ANCHOR_IDS:
        lm = face_landmarks[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)

def apply_color(frame, mask, color_bgr):
    if color_bgr is None:
        return frame

    # mask is now a float 0.0-1.0 (soft edges)
    mask_smooth = np.clip(mask, 0, 1)

    # Convert to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    color_pixel = np.uint8([[list(color_bgr)]])
    color_hsv = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)

    target_hue = color_hsv[0]
    target_sat = color_hsv[1]

    # Specular highlight detection — preserve shiny spots
    value_channel = frame_hsv[:, :, 2]
    specular_mask = (value_channel > 200).astype(np.float32)
    specular_mask = cv2.GaussianBlur(specular_mask, (7, 7), 0)
    non_specular = 1.0 - specular_mask

    effective_mask = mask_smooth * non_specular

    output_hsv = frame_hsv.copy()

    # Replace hue
    output_hsv[:, :, 0] = (frame_hsv[:, :, 0] * (1 - effective_mask) +
                            target_hue * effective_mask)

    # Boost saturation
    value_map = frame_hsv[:, :, 2] / 255.0
    sat_boost = target_sat * (0.5 + 0.5 * value_map)
    output_hsv[:, :, 1] = np.clip(
        frame_hsv[:, :, 1] * (1 - effective_mask) + sat_boost * effective_mask,
        0, 255
    )

    output_bgr = cv2.cvtColor(output_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Final blend with soft mask
    mask_3ch = cv2.merge([effective_mask, effective_mask, effective_mask])
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