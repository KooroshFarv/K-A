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

def build_lut(color_bgr, hair_type="dark"):
    """
    Pre-calculate a lookup table (LUT) for hair color rendering.
    Maps every possible pixel brightness (0-255) to the correct
    colored output — this is how Banuba achieves realistic results.
    
    Two LUTs are built based on hair type:
    - Dark hair LUT: lifts shadows so colors show on dark base
    - Light hair LUT: preserves brightness, adjusts hue/saturation
    
    At render time, we just do a table lookup instead of per-pixel math.
    This is faster AND more accurate than real-time HSV calculations.
    """
    if color_bgr is None:
        return None

    # Convert target color to HSV
    color_pixel = np.uint8([[list(color_bgr)]])
    color_hsv = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    target_hue = color_hsv[0]
    target_sat = color_hsv[1]
    target_val = color_hsv[2]

    # Build a 256-entry LUT
    # Each entry maps input brightness -> output BGR color
    lut = np.zeros((256, 3), dtype=np.float32)

    for i in range(256):
        brightness = i / 255.0  # normalize to 0-1

        # --- Hue: always use target color's hue ---
        out_hue = target_hue

        if brightness > 0.78:
            out_sat = target_sat * (1.0 - brightness) * 2.5
        else:
            if hair_type == "dark" and target_val > 150:
               out_sat = target_sat * (0.6 + 0.4 * brightness)
            else:
                out_sat = target_sat * (0.4 + 0.6 * brightness)   

        if hair_type == "dark" and target_val > 150:
            bleach_strength = min((target_val - 150) / 105.0, 0.55)
            floor_lift = bleach_strength * 0.35
            lifted = brightness * (1.0 - floor_lift) + floor_lift
            out_val = lifted * 0.82 + (target_val / 255.0) * 0.18
            out_val = min(out_val, 1.0)

        elif hair_type == "light" and target_val < 80:
            darken_strength = (80 - target_val) / 80.0
            darkened_base = brightness * (1.0 - darken_strength * 0.5)
            out_val = darkened_base * 0.8 + (target_val / 255.0) * 0.2
        elif hair_type == "medium" and target_val > 180:
            bleach_strength = (target_val - 180) / 75.0 * 0.6  # gentler lift
            bleached_base = 0.5 + bleach_strength * 0.2
            lifted = brightness * (1.0 - bleach_strength) + bleached_base * bleach_strength
            out_val = lifted * 0.8 + (target_val / 255.0) * 0.2    
        else:
            # Normal case
            out_val = brightness * 0.85 + (target_val / 255.0) * 0.15

        # Clamp all values
        out_hue = np.clip(out_hue, 0, 179)
        out_sat = np.clip(out_sat, 0, 255)
        out_val = np.clip(out_val * 255, 0, 255)

        # Convert HSV -> BGR for this LUT entry
        hsv_pixel = np.uint8([[[out_hue, out_sat, out_val]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        lut[i] = bgr.astype(np.float32)

    return lut


def apply_lut(frame, mask, lut):
    if lut is None:
        return frame

    mask_smooth = np.clip(mask, 0, 1)

    # Double-blur the mask for extra soft edges
    # First blur already happened in get_hair_mask (31px)
    # Second pass here softens the transition even further
    mask_smooth = cv2.GaussianBlur(mask_smooth, (25, 25), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = lut[gray]

    # Specular highlight mask
    specular = (gray > 200).astype(np.float32)
    specular = cv2.GaussianBlur(specular, (7, 7), 0)
    non_specular = 1.0 - specular

    effective_mask = mask_smooth * non_specular

    # Reduce opacity at soft edge areas — pixels with mask 0.0-0.5
    # get proportionally less color to avoid harsh transitions
    edge_factor = np.where(effective_mask < 0.5,
                           effective_mask * 1.6,  # softer at edges
                           effective_mask)
    edge_factor = np.clip(edge_factor, 0, 1)

    mask_3ch = cv2.merge([edge_factor, edge_factor, edge_factor])

    result = (frame.astype(np.float32) * (1 - mask_3ch * 0.82) +
              colored * (mask_3ch * 0.88))

    return result.clip(0, 255).astype(np.uint8)



def detect_ambient_light(frame, mask):
    """
    Detect ambient lighting color and intensity from the background/skin areas.
    Returns a lighting correction factor to apply to hair color rendering.
    
    Salons have varying lighting — warm incandescent, cool LED, natural daylight.
    Each changes how hair color appears. This function samples the scene's
    ambient light and adjusts the LUT accordingly so colors look correct
    under any salon lighting condition.
    """
    h, w = frame.shape[:2]


    bg_samples = []

    corner_size = min(h, w) // 8
    regions = [
        frame[0:corner_size, 0:corner_size],
        frame[0:corner_size, w-corner_size:w],
        frame[h-corner_size:h, 0:corner_size],
        frame[h-corner_size:h, w-corner_size:w],
        frame[0:corner_size, w//4:3*w//4],
    ]

    for region in regions:
        if region.size > 0:
            bg_samples.append(region.reshape(-1, 3))

    if not bg_samples:
        return 1.0, 0, 0  # no adaptation

    bg_pixels = np.vstack(bg_samples).astype(np.float32)

    bg_lab = cv2.cvtColor(
        bg_pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2Lab
    ).reshape(-1, 3).astype(np.float32)


    avg_L = np.median(bg_lab[:, 0]) 
    avg_a = np.median(bg_lab[:, 1])
    avg_b = np.median(bg_lab[:, 2]) 


    brightness_factor = np.clip(avg_L / 60.0, 0.7, 1.3)  # dim vs bright room
    ws_val = np.clip((avg_b - 128) / 20.0, -1.0, 1.0)  # warm/cool shift
    green_shift = np.clip((avg_a - 128) / 20.0, -1.0, 1.0)   # green/magenta cast

    return brightness_factor, ws_val, green_shift


def build_lut_with_lighting(color_bgr, hair_type="dark",
                             brightness_factor=1.0, ws_val=0.0, green_shift=0.0):
    """
    Extended LUT builder that incorporates ambient lighting correction.
    Same as build_lut but adjusts the output colors based on scene lighting.
    
    Under warm salon lights: colors shift slightly warmer
    Under cool LED lights: colors shift slightly cooler
    Under dim lighting: colors are slightly lifted to compensate
    """
    if color_bgr is None:
        return None

    color_pixel = np.uint8([[list(color_bgr)]])
    color_hsv = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    target_hue = color_hsv[0]
    target_sat = color_hsv[1]
    target_val = color_hsv[2]


    hue_shift = ws_val * 8.0
    target_hue = np.clip(target_hue + hue_shift, 0, 179)

    sat_adjust = brightness_factor * 0.95
    target_sat = np.clip(target_sat * sat_adjust, 0, 255)

    lut = np.zeros((256, 3), dtype=np.float32)

    for i in range(256):
        brightness = i / 255.0

        out_hue = target_hue

        if brightness > 0.78:
            out_sat = target_sat * (1.0 - brightness) * 2.5
        else:
            if hair_type == "dark" and target_val > 150:
                out_sat = target_sat * (0.6 + 0.4 * brightness)
            else:
                out_sat = target_sat * (0.4 + 0.6 * brightness)

        if hair_type == "dark" and target_val > 150:
            bleach_strength = min((target_val - 150) / 105.0, 0.55)
            floor_lift = bleach_strength * 0.35
            lifted = brightness * (1.0 - floor_lift) + floor_lift
            out_val = lifted * 0.82 + (target_val / 255.0) * 0.18
            out_val = min(out_val, 1.0)


        elif hair_type == "light" and target_val < 80:
            darken_strength = (80 - target_val) / 80.0
            darkened_base = brightness * (1.0 - darken_strength * 0.5)
            out_val = darkened_base * 0.8 + (target_val / 255.0) * 0.2
        elif hair_type == "medium" and target_val > 180:
            bleach_strength = (target_val - 180) / 75.0 * 0.6
            bleached_base = 0.5 + bleach_strength * 0.2
            lifted = brightness * (1.0 - bleach_strength) + bleached_base * bleach_strength
            out_val = lifted * 0.8 + (target_val / 255.0) * 0.2
        else:
            out_val = brightness * 0.85 + (target_val / 255.0) * 0.15

        out_val = np.clip(out_val * brightness_factor, 0, 1.0)

        out_hue = np.clip(out_hue, 0, 179)
        out_sat = np.clip(out_sat, 0, 255)
        out_val = np.clip(out_val * 255, 0, 255)

        hsv_pixel = np.uint8([[[out_hue, out_sat, out_val]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        lut[i] = bgr.astype(np.float32)

    return lut


def apply_color(frame, mask, color_bgr, hair_type="medium"):
    """
    Main color application — builds LUT then applies it.
    Called every frame but LUT should be cached externally
    for better performance (rebuild only when color changes).
    """
    if color_bgr is None:
        return frame

    lut = build_lut(color_bgr, hair_type)
    return apply_lut(frame, mask, lut)


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}


@sock.route('/stream')
def stream(ws):
    print("Client connected to stream")
    landmarker = vision.FaceLandmarker.create_from_options(face_options)

    base_mask = None
    base_anchors = None
    base_gray = None
    frame_count = 0
    current_color = None
    smooth_matrix = None
    SMOOTH_FACTOR = 0.85
    current_lut = None
    last_color = None
    hair_type = "medium"

    while True:
        try:
            raw = ws.receive()
            if raw is None:
                break
            msg = json.loads(raw)

            if "color_bgr" in msg:
                current_color = tuple(msg["color_bgr"]) if msg["color_bgr"] else None
                if current_color and base_mask is not None and 'frame' in dir():
                    bf, ws_val, gs = detect_ambient_light(frame, base_mask)
                    current_lut = build_lut_with_lighting(current_color, hair_type, bf, ws_val, gs)
                elif current_color:
                    current_lut = build_lut(current_color, hair_type)
                else:
                    current_lut = None
                last_color = current_color        

            if "frame" not in msg:
                continue

            

            frame = decode_frame(msg["frame"])
            h, w = frame.shape[:2]
            frame_count += 1


            if frame_count % 30 == 0 and base_mask is not None and current_color is not None:
                bf, ws_val, gs = detect_ambient_light(frame, base_mask)
                current_lut = build_lut_with_lighting(current_color, hair_type, bf, ws_val, gs)  



            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                current_anchors = get_anchors(face_landmarks, w, h)
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                need_reseg = (base_mask is None) or (frame_count % 60 == 0)

                if need_reseg:
                    # Full re-segmentation
                    base_mask = get_hair_mask(frame)
                    base_anchors = current_anchors.copy()
                    base_gray = current_gray.copy()
                    smooth_matrix = None
                    display_mask = base_mask

                    # Update hair type and rebuild LUT
                    hair_pixels = frame[base_mask > 0.3]
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
                    if current_color:
                        bf, ws_val, gs = detect_ambient_light(frame, base_mask)
                        current_lut = build_lut_with_lighting(current_color, hair_type, bf, ws_val, gs)

                else:
                    warped_mask = base_mask  # fallback

                    if base_gray is not None:
                        anchor_pts = base_anchors.reshape(-1, 1, 2).astype(np.float32)

                        tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                            base_gray,          # previous frame (grayscale)
                            current_gray,       # current frame (grayscale)
                            anchor_pts,         # points to track
                            None,               # output points
                            winSize=(21, 21),   # search window size
                            maxLevel=3,         # pyramid levels (handles large movements)
                            criteria=(
                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                20, 0.01
                            )
                        )

                        # Filter to only well-tracked points (status == 1)
                        good_base = anchor_pts[status == 1]
                        good_tracked = tracked_pts[status == 1]

                        if len(good_base) >= 3:
                            # Enough points tracked — compute affine transform
                            # from optical flow results (more accurate than landmarks alone)
                            transform_matrix, _ = cv2.estimateAffinePartial2D(
                                good_base, good_tracked
                            )

                            if transform_matrix is not None:
                                # Smooth the transform to reduce jitter
                                if smooth_matrix is None:
                                    smooth_matrix = transform_matrix
                                else:
                                    smooth_matrix = (
                                        SMOOTH_FACTOR * smooth_matrix +
                                        (1 - SMOOTH_FACTOR) * transform_matrix
                                    )
                                warped_mask = cv2.warpAffine(
                                    base_mask, smooth_matrix, (w, h)
                                )
                        else:
                            # Fallback: use face landmark affine transform
                            transform_matrix, _ = cv2.estimateAffinePartial2D(
                                base_anchors, current_anchors
                            )
                            if transform_matrix is not None:
                                if smooth_matrix is None:
                                    smooth_matrix = transform_matrix
                                else:
                                    smooth_matrix = (
                                        SMOOTH_FACTOR * smooth_matrix +
                                        (1 - SMOOTH_FACTOR) * transform_matrix
                                    )
                                warped_mask = cv2.warpAffine(
                                    base_mask, smooth_matrix, (w, h)
                                )
                        
                        base_gray = current_gray.copy()
                        base_anchors = current_anchors.copy()

                    display_mask = warped_mask

                # Apply color using cached LUT
                if current_lut is not None:
                    output = apply_lut(frame, display_mask, current_lut)
                else:
                    output = frame
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