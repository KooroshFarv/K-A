import cv2
import requests
import base64
import numpy as np
import threading
import json
import websocket
import time

SERVER_URL = "http://localhost:5050"
WS_URL = "ws://localhost:5050/stream"

HAIRSTYLES = [
    "bob cut", "long wavy hair", "short pixie cut", "curly afro",
    "straight long hair", "undercut fade", "long dreadlocks", "buzz cut",
]

PRESETS = [
    ("Black",   (20, 20, 20)),
    ("Brown",   (30, 80, 130)),
    ("Blonde",  (80, 200, 255)),
    ("Auburn",  (20, 60, 160)),
    ("Gray",    (160, 160, 160)),
    ("White",   (240, 240, 240)),
]

current_color_bgr = None
current_color_name = "Original"
status_msg = "Connecting..."
status_color = (255, 255, 255)
mode = "live"
result_frame = None       # streamed live color frame
hairstyle_result = None   # static generated hairstyle frame
processing = False
last_frame = None
ws_connected = False

SPECTRUM_X = 10
SPECTRUM_H = 28
SPECTRUM_W = 500

def build_spectrum(w=500, h=28):
    bar = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        hue = int(x / w * 180)
        hsv_pixel = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        bar[:, x] = bgr
    return bar

spectrum_img = build_spectrum(SPECTRUM_W, SPECTRUM_H)

def get_color_from_spectrum(x):
    x = max(0, min(x - SPECTRUM_X, SPECTRUM_W - 1))
    return tuple(int(c) for c in spectrum_img[SPECTRUM_H // 2, x])

def encode_frame(frame, quality=60):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

def decode_frame(frame_b64):
    frame_bytes = base64.b64decode(frame_b64)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --- WebSocket streaming thread ---
ws_app = None

def on_ws_message(ws, message):
    global result_frame
    try:
        data = json.loads(message)
        result_frame = decode_frame(data["frame"])
    except Exception as e:
        print(f"WS message error: {e}")

def on_ws_open(ws):
    global ws_connected, status_msg, status_color
    ws_connected = True
    status_msg = "Live mirror ready"
    status_color = (0, 255, 150)
    print("WebSocket connected")

def on_ws_close(ws, *args):
    global ws_connected, status_msg, status_color
    ws_connected = False
    status_msg = "Disconnected - reconnecting..."
    status_color = (0, 0, 255)
    print("WebSocket closed")

def on_ws_error(ws, error):
    print(f"WebSocket error: {error}")

def start_websocket():
    global ws_app
    while True:
        try:
            ws_app = websocket.WebSocketApp(
                WS_URL,
                on_message=on_ws_message,
                on_open=on_ws_open,
                on_close=on_ws_close,
                on_error=on_ws_error
            )
            ws_app.run_forever()
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
        time.sleep(2)  # retry delay

def send_frame_to_stream(frame):
    """Called continuously from main loop to stream frames"""
    if ws_connected and ws_app and ws_app.sock:
        try:
            payload = {
                "frame": encode_frame(frame),
                "color_bgr": list(current_color_bgr) if current_color_bgr else None
            }
            ws_app.send(json.dumps(payload))
        except Exception:
            pass

def request_hairstyle(frame, hairstyle_name):
    global hairstyle_result, mode, status_msg, status_color, processing
    try:
        status_msg = f"Generating {hairstyle_name}..."
        status_color = (0, 200, 255)
        response = requests.post(
            f"{SERVER_URL}/hairstyle",
            json={"frame": encode_frame(frame, quality=85), "hairstyle": hairstyle_name},
            timeout=120
        )
        if response.status_code == 200:
            hairstyle_result = decode_frame(response.json()["frame"])
            mode = "result"
            status_msg = f"Showing: {hairstyle_name}"
            status_color = (0, 255, 150)
        else:
            status_msg = "Generation failed"
            status_color = (0, 0, 255)
    except Exception as e:
        status_msg = f"Error: {str(e)[:30]}"
        status_color = (0, 0, 255)
    processing = False

def mouse_click(event, x, y, flags, param):
    global current_color_bgr, current_color_name, processing, last_frame, mode
    if event != cv2.EVENT_LBUTTONDOWN or last_frame is None:
        return

    h, w = last_frame.shape[:2]
    bottom_panel_y = h - 180

    spec_y = bottom_panel_y + 45
    if spec_y <= y <= spec_y + SPECTRUM_H and SPECTRUM_X <= x <= SPECTRUM_X + SPECTRUM_W:
        current_color_bgr = get_color_from_spectrum(x)
        current_color_name = "Custom"
        mode = "live"
        return

    preset_y = bottom_panel_y + 85
    preset_btn_w = 80
    for i, (name, color) in enumerate(PRESETS):
        bx = SPECTRUM_X + i * (preset_btn_w + 6)
        if bx <= x <= bx + preset_btn_w and preset_y <= y <= preset_y + 32:
            current_color_bgr = color
            current_color_name = name
            mode = "live"
            return

    orig_x = SPECTRUM_X + len(PRESETS) * (preset_btn_w + 6)
    if orig_x <= x <= orig_x + preset_btn_w and preset_y <= y <= preset_y + 32:
        current_color_bgr = None
        current_color_name = "Original"
        mode = "live"
        return

    if not processing:
        hs_w, hs_h = 115, 34
        hs_x_start = w - 4 * (hs_w + 8) - 10
        hs_y_start = bottom_panel_y + 10
        for i, style in enumerate(HAIRSTYLES):
            col = i % 4
            row = i // 4
            bx = hs_x_start + col * (hs_w + 8)
            by = hs_y_start + row * (hs_h + 6)
            if bx <= x <= bx + hs_w and by <= y <= by + hs_h:
                processing = True
                threading.Thread(target=request_hairstyle, args=(last_frame.copy(), style)).start()
                return

    if mode == "result":
        if 10 <= x <= 120 and 80 <= y <= 118:
            mode = "live"

def draw_ui(frame):
    h, w = frame.shape[:2]
    ui = frame.copy()

    cv2.rectangle(ui, (0, 0), (w, 65), (15, 15, 15), -1)
    cv2.putText(ui, "ANGELS", (20, 46), cv2.FONT_HERSHEY_DUPLEX, 1.3, (200, 160, 255), 2)

    cv2.putText(ui, status_msg, (160, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 1)

    live_dot_color = (0, 255, 0) if ws_connected else (0, 0, 255)
    cv2.circle(ui, (w - 30, 32), 8, live_dot_color, -1)
    cv2.putText(ui, "LIVE" if ws_connected else "OFF", (w - 90, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, live_dot_color, 1)

    if processing:
        cv2.putText(ui, "Generating...", (w - 250, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    bottom_panel_y = h - 180

    cv2.rectangle(ui, (0, bottom_panel_y), (w, h), (15, 15, 15), -1)
    cv2.line(ui, (0, bottom_panel_y), (w, bottom_panel_y), (50, 50, 50), 1)

    spec_y = bottom_panel_y + 45
    cv2.putText(ui, "HAIR COLOR (LIVE)", (SPECTRUM_X, bottom_panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    ui[spec_y:spec_y + SPECTRUM_H, SPECTRUM_X:SPECTRUM_X + SPECTRUM_W] = spectrum_img

    cv2.rectangle(ui, (SPECTRUM_X, spec_y), (SPECTRUM_X + SPECTRUM_W, spec_y + SPECTRUM_H),
                  (100, 100, 100), 1)

    if current_color_name == "Custom" and current_color_bgr:
        bgr = current_color_bgr
        hue_val = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0][0][0]
        dot_x = SPECTRUM_X + int(hue_val / 180 * SPECTRUM_W)
        cv2.circle(ui, (dot_x, spec_y + SPECTRUM_H // 2), 8, (255, 255, 255), 2)

    preset_y = bottom_panel_y + 85
    preset_btn_w = 80
    cv2.putText(ui, "NATURAL TONES", (SPECTRUM_X, preset_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    all_presets = PRESETS + [("Original", None)]
    for i, (name, color) in enumerate(all_presets):
        bx = SPECTRUM_X + i * (preset_btn_w + 6)
        is_selected = name == current_color_name
        border_color = (200, 160, 255) if is_selected else (70, 70, 70)
        cv2.rectangle(ui, (bx, preset_y), (bx + preset_btn_w, preset_y + 32), border_color,
                      2 if is_selected else 1)
        if color:
            cv2.rectangle(ui, (bx + 2, preset_y + 2), (bx + 18, preset_y + 30), color, -1)
        cv2.putText(ui, name, (bx + (4 if color else 8), preset_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 160, 255) if is_selected else (200, 200, 200), 1)

    hs_w, hs_h = 115, 34
    hs_x_start = w - 4 * (hs_w + 8) - 10
    hs_y_start = bottom_panel_y + 10

    cv2.putText(ui, "HAIRSTYLE TRY-ON", (hs_x_start, bottom_panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    for i, style in enumerate(HAIRSTYLES):
        col = i % 4
        row = i // 4
        bx = hs_x_start + col * (hs_w + 8)
        by = hs_y_start + row * (hs_h + 6)
        cv2.rectangle(ui, (bx, by), (bx + hs_w, by + hs_h), (70, 70, 70), 1)
        cv2.putText(ui, style[:14], (bx + 5, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1)

    if mode == "result":
        cv2.rectangle(ui, (10, 75), (120, 115), (40, 40, 40), -1)
        cv2.rectangle(ui, (10, 75), (120, 115), (100, 100, 100), 1)
        cv2.putText(ui, "< BACK", (18, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 160, 255), 1)

    cv2.putText(ui, "Click spectrum for live color | Click hairstyle to try on | Q: quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    return ui


# --- Start WebSocket thread ---
ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Angels mirror app running...")
print("Make sure cloud_api.py is running in another terminal!")

cv2.namedWindow('Angels - AI Hair Studio')
cv2.setMouseCallback('Angels - AI Hair Studio', mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    last_frame = frame.copy()
    h, w = frame.shape[:2]

    if mode == "live":
        send_frame_to_stream(frame)  # stream every frame for live color preview
        if result_frame is not None and ws_connected:
            display = cv2.resize(result_frame, (w, h))
        else:
            display = frame
        display = draw_ui(display)
    elif mode == "result" and hairstyle_result is not None:
        display = cv2.resize(hairstyle_result, (w, h))
        display = draw_ui(display)
    else:
        display = draw_ui(frame)

    cv2.imshow('Angels - AI Hair Studio', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()