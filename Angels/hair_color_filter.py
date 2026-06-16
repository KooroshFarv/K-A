import cv2
import torch
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

print("Loading model... this may take a minute the first time.")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model.eval()

HAIR_LABEL = 2

# --- Performance settings ---
PROCESS_SIZE = 256
FRAME_SKIP = 4

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Color options (BGR format) ---
COLORS = {
    ord('1'): ("Red",    (0, 0, 255)),
    ord('2'): ("Blue",   (255, 0, 0)),
    ord('3'): ("Green",  (0, 255, 0)),
    ord('4'): ("Purple", (200, 0, 200)),
    ord('5'): ("Blonde", (80, 200, 255)),
    ord('6'): ("Pink",   (180, 80, 255)),
    ord('0'): ("None",   None),
}

current_color_name, current_color = "Red", (0, 0, 255)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened successfully.")
print("Press 1-6 to change hair color, 0 for no filter, 'q' to quit.")

frame_count = 0
hair_mask = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    h, w = frame.shape[:2]
    frame_count += 1

    if frame_count % FRAME_SKIP == 0 or hair_mask is None:
        small = cv2.resize(frame, (PROCESS_SIZE, PROCESS_SIZE))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_small)

        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].numpy()
        hair_mask = (pred == HAIR_LABEL).astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, fw, fh) in faces:
            pad_x = int(fw * 0.1)
            pad_y = int(fh * 0.05)
            x1, y1 = max(0, x + pad_x), max(0, y + pad_y)
            x2, y2 = min(w, x + fw - pad_x), min(h, y + fh)
            hair_mask[y1:y2, x1:x2] = 0

    output = frame.copy()

    if current_color is not None:
        color_layer = np.zeros_like(frame)
        color_layer[:] = current_color

        mask_3ch = cv2.merge([hair_mask, hair_mask, hair_mask]).astype(np.float32)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray_frame, gray_frame, gray_frame]).astype(np.float32) / 255.0

        tinted = (color_layer.astype(np.float32) * gray_3ch * 1.3).clip(0, 255)

        output = (output.astype(np.float32) * (1 - mask_3ch) + tinted * mask_3ch).astype(np.uint8)

    cv2.putText(output, f"Color: {current_color_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "Press 1-6 to change color, 0 for none, q to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Angels - Hair Color Filter', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in COLORS:
        current_color_name, current_color = COLORS[key]

cap.release()
cv2.destroyAllWindows()