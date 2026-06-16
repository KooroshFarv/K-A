import cv2
import requests
import os
import time

API_KEY = os.environ.get("LIGHTX_API_KEY")

HAIRSTYLES = {
    ord('1'): "bob cut",
    ord('2'): "long wavy hair",
    ord('3'): "short pixie cut",
    ord('4'): "curly afro",
    ord('5'): "straight long hair",
    ord('6'): "undercut fade",
    ord('7'): "long dreadlocks",
    ord('8'): "buzz cut",
    ord('0'): None,
}

def generate_hairstyle(photo_path, hairstyle_name):
    with open(photo_path, "rb") as f:
        upload_response = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": f}
        )

    upload_data = upload_response.json()
    image_url = upload_data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
    print(f"Image uploaded: {image_url}")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    payload = {
        "imageUrl": image_url,
        "textPrompt": hairstyle_name,
    }

    response = requests.post(
        "https://api.lightxeditor.com/external/api/v1/hairstyle",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    data = response.json()
    order_id = data["body"]["orderId"]
    print(f"Order ID: {order_id}. Waiting for result...")

    for i in range(20):
        time.sleep(5)
        print(f"Checking result... attempt {i+1}")

        poll_response = requests.post(
            "https://api.lightxeditor.com/external/api/v1/order-status",
            headers=headers,
            json={"orderId": order_id}
        )

        poll_data = poll_response.json()
        print(f"Poll response: {poll_data}")

        status = poll_data.get("body", {}).get("status")
        if status == "active":
            result_url = poll_data["body"].get("output")
            if result_url:
                img_response = requests.get(result_url)
                output_path = f"result_{hairstyle_name.replace(' ', '_')}.jpg"
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                return output_path

    raise Exception("Timed out waiting for result")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened. Press 's' to capture photo, then 1-8 for hairstyles, 'q' to quit.")

captured_photo = None
result_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    display = frame.copy()

    cv2.putText(display, "Press 's' to capture photo", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if captured_photo:
        cv2.putText(display, "Photo ready! Press 1-8 for hairstyle", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(display, "1:Bob 2:Wavy 3:Pixie 4:Afro 5:Straight 6:Undercut 7:Dreads 8:Buzz",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Angels - Hairstyle Generator', display)

    if result_image is not None:
        cv2.imshow('Angels - Result', result_image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        photo_path = "temp_photo.jpg"
        cv2.imwrite(photo_path, frame)
        captured_photo = photo_path
        print("Photo captured!")
    elif captured_photo and key in HAIRSTYLES:
        hairstyle = HAIRSTYLES[key]
        if hairstyle:
            print(f"Generating {hairstyle}... please wait 10-30 seconds")
            try:
                output_path = generate_hairstyle(captured_photo, hairstyle)
                result_image = cv2.imread(output_path)
                if result_image is not None:
                    print("Done! Showing result.")
                else:
                    print("Error: Could not load result image")
            except Exception as e:
                print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()