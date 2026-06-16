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

PROCESS_SIZE = 256   
FRAME_SKIP = 4          

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

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

    overlay = frame.copy()
    overlay[hair_mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    cv2.imshow('Angels - Hair Detection', blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()