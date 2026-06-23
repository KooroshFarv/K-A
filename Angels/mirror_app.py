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

COLOR_FAMILIES = [
    ("Black",     (26, 26, 26)),
    ("Brown",     (42, 58, 107)),
    ("Blonde",    (67, 168, 212)),
    ("Auburn",    (26, 58, 139)),
    ("Copper",    (30, 90, 185)),
    ("Red",       (43, 57, 192)),
    ("Rose Gold", (145, 155, 220)),
    ("Pink",      (140, 30, 233)),
    ("Purple",    (190, 47, 123)),
    ("Blue",      (192, 101, 21)),
    ("Teal",      (130, 160, 20)),
    ("Green",     (30, 160, 30)),
    ("Orange",    (20, 120, 220)),
    ("Gray",      (158, 158, 158)),
    ("White",     (245, 245, 245)),
]

COLOR_SHADES = {
    "Black": [
        ("Jet Black",        (10, 10, 10)),
        ("Blue Black",       (20, 10, 8)),
        ("Soft Black",       (30, 25, 22)),
        ("Natural Black",    (25, 20, 18)),
        ("Off Black",        (42, 38, 35)),
        ("Blue-Black Noir",  (15, 8, 5)),
    ],
    "Brown": [
        ("Espresso",         (3, 21, 44)),
        ("Dark Choc",        (13, 31, 62)),
        ("Chestnut",         (40, 60, 120)),
        ("Medium Brown",     (48, 74, 122)),
        ("Warm Chestnut",    (35, 65, 140)),
        ("Caramel Brown",    (60, 100, 160)),
        ("Light Brown",      (74, 102, 160)),
        ("Mushroom",         (90, 110, 145)),
        ("Cendre Brown",     (85, 100, 130)),
        ("Mochaccino",       (30, 55, 100)),
        ("Truffle",          (45, 70, 110)),
        ("Toffee",           (55, 90, 155)),
    ],
    "Blonde": [
        ("Platinum",         (210, 230, 240)),
        ("Pearl Blonde",     (195, 220, 235)),
        ("Ash Blonde",       (169, 197, 212)),
        ("Champagne",        (150, 195, 220)),
        ("Dark Ash",         (130, 165, 190)),
        ("Sandy Blonde",     (120, 175, 210)),
        ("Honey Blonde",     (80, 175, 215)),
        ("Golden Blonde",    (60, 160, 210)),
        ("Warm Golden",      (45, 140, 195)),
        ("Caramel Blonde",   (70, 150, 200)),
        ("Dirty Blonde",     (100, 145, 175)),
        ("Strawberry",       (105, 145, 225)),
    ],
    "Auburn": [
        ("Light Auburn",     (70, 110, 195)),
        ("Golden Auburn",    (55, 100, 185)),
        ("Auburn",           (45, 80, 160)),
        ("Deep Auburn",      (30, 60, 130)),
        ("Russet",           (35, 80, 150)),
        ("Mahogany",         (32, 55, 105)),
        ("Warm Mahogany",    (40, 65, 120)),
        ("Rich Auburn",      (28, 55, 125)),
        ("Dark Auburn",      (20, 45, 110)),
        ("Chestnut Auburn",  (50, 85, 165)),
    ],
    "Copper": [
        ("Light Copper",     (65, 130, 225)),
        ("Copper",           (51, 115, 200)),
        ("Golden Copper",    (40, 110, 210)),
        ("Rose Copper",      (100, 120, 210)),
        ("Bright Copper",    (35, 100, 220)),
        ("Deep Copper",      (30, 85, 180)),
        ("Copper Brown",     (40, 90, 165)),
        ("Dark Copper",      (25, 70, 150)),
    ],
    "Red": [
        ("Light Red",        (60, 80, 220)),
        ("Bright Red",       (50, 50, 210)),
        ("Crimson",          (43, 57, 192)),
        ("Cherry Red",       (38, 50, 169)),
        ("Deep Red",         (33, 43, 146)),
        ("Dark Red",         (25, 35, 130)),
        ("Burgundy",         (35, 20, 110)),
        ("Deep Burgundy",    (20, 12, 85)),
        ("Mahogany Red",     (30, 40, 140)),
        ("Garnet",           (25, 25, 105)),
        ("Wine Red",         (30, 15, 100)),
        ("Ruby Red",         (40, 30, 160)),
    ],
    "Rose Gold": [
        ("Light Rose Gold",  (170, 175, 235)),
        ("Rose Gold",        (145, 155, 220)),
        ("Warm Rose Gold",   (130, 145, 215)),
        ("Deep Rose Gold",   (110, 130, 200)),
        ("Copper Rose",      (120, 140, 210)),
        ("Blush Gold",       (155, 165, 230)),
        ("Dusty Rose Gold",  (135, 140, 205)),
        ("Pink Gold",        (140, 150, 225)),
    ],
    "Pink": [
        ("Pastel Pink",      (195, 182, 255)),
        ("Baby Pink",        (185, 167, 244)),
        ("Rose Gold Pink",   (160, 155, 240)),
        ("Dusty Rose",       (130, 130, 210)),
        ("Bubblegum",        (160, 140, 255)),
        ("Hot Pink",         (100, 30, 233)),
        ("Rose",             (130, 30, 225)),
        ("Fuchsia",          (88, 24, 194)),
        ("Deep Pink",        (70, 20, 180)),
        ("Magenta",          (80, 25, 200)),
        ("Electric Pink",    (60, 15, 220)),
        ("Flamingo",         (150, 120, 245)),
    ],
    "Purple": [
        ("Lavender",         (215, 157, 179)),
        ("Lilac",            (205, 155, 195)),
        ("Mauve",            (175, 120, 175)),
        ("Amethyst",         (175, 89, 155)),
        ("Violet",           (155, 31, 123)),
        ("Purple",           (150, 47, 123)),
        ("Deep Violet",      (120, 25, 100)),
        ("Plum",             (90, 27, 106)),
        ("Deep Plum",        (65, 15, 85)),
        ("Eggplant",         (50, 12, 70)),
        ("Grape",            (100, 30, 110)),
        ("Orchid",           (160, 80, 160)),
    ],
    "Blue": [
        ("Ice Blue",         (235, 200, 120)),
        ("Baby Blue",        (230, 185, 100)),
        ("Sky Blue",         (220, 170, 80)),
        ("Ocean Blue",       (200, 140, 40)),
        ("Cobalt",           (185, 120, 20)),
        ("Sapphire",         (170, 100, 10)),
        ("Royal Blue",       (155, 80, 5)),
        ("Midnight Blue",    (70, 20, 5)),
        ("Navy Blue",        (50, 10, 5)),
        ("Denim Blue",       (140, 100, 30)),
        ("Petrol Blue",      (110, 90, 15)),
        ("Electric Blue",    (200, 150, 0)),
    ],
    "Teal": [
        ("Light Teal",       (180, 180, 10)),
        ("Mint Teal",        (160, 190, 20)),
        ("Teal",             (130, 160, 10)),
        ("Deep Teal",        (100, 130, 5)),
        ("Emerald Teal",     (80, 140, 10)),
        ("Dark Teal",        (60, 100, 5)),
        ("Petrol Teal",      (90, 110, 10)),
        ("Teal Blue",        (120, 150, 15)),
    ],
    "Green": [
        ("Mint Green",       (140, 210, 80)),
        ("Pastel Green",     (120, 200, 70)),
        ("Forest Green",     (30, 120, 20)),
        ("Emerald",          (20, 150, 30)),
        ("Olive Green",      (30, 110, 50)),
        ("Dark Green",       (10, 90, 15)),
        ("Lime Green",       (50, 200, 50)),
        ("Sage Green",       (80, 150, 80)),
    ],
    "Orange": [
        ("Peach",            (120, 160, 240)),
        ("Light Orange",     (80, 140, 240)),
        ("Orange",           (30, 120, 230)),
        ("Tangerine",        (20, 110, 225)),
        ("Burnt Orange",     (15, 90, 195)),
        ("Pumpkin",          (10, 100, 210)),
        ("Deep Orange",      (5, 80, 185)),
        ("Amber",            (40, 130, 220)),
    ],
    "Gray": [
        ("Silver",           (200, 200, 200)),
        ("Pearl Gray",       (190, 190, 195)),
        ("Ash Gray",         (178, 180, 182)),
        ("Cool Gray",        (165, 165, 170)),
        ("Steel Gray",       (140, 140, 145)),
        ("Slate Gray",       (120, 120, 128)),
        ("Salt & Pepper",    (140, 140, 140)),
        ("Graphite",         (90, 90, 95)),
        ("Charcoal",         (65, 65, 70)),
        ("Smoke",            (210, 210, 210)),
    ],
    "White": [
        ("Pure White",       (255, 255, 255)),
        ("Snow White",       (250, 250, 255)),
        ("Pearl White",      (240, 240, 245)),
        ("Platinum",         (230, 228, 226)),
        ("Icy White",        (235, 238, 245)),
        ("Creamy White",     (225, 225, 215)),
    ],
}

# UI state
current_color_bgr = None
current_color_name = "Original"
status_msg = "Connecting..."
status_color = (255, 255, 255)
mode = "live"           # live, result
ui_state = "families"   # families, shades
selected_family = None
result_frame = None
hairstyle_result = None
processing = False
last_frame = None
ws_connected = False
ws_app = None

# Layout constants
BOTTOM_H = 200
CIRCLE_R = 28
CIRCLE_GAP = 14

def build_family_positions(w, h):
    """Calculate positions for color family circles"""
    n = len(COLOR_FAMILIES)
    total_w = n * (CIRCLE_R * 2 + CIRCLE_GAP) - CIRCLE_GAP
    start_x = (w - total_w) // 2
    y = h - BOTTOM_H + 60
    positions = []
    for i in range(n):
        cx = start_x + i * (CIRCLE_R * 2 + CIRCLE_GAP) + CIRCLE_R
        positions.append((cx, y))
    return positions

def build_shade_positions(w, h, family):
    """Calculate positions for shade circles"""
    shades = COLOR_SHADES[family]
    n = len(shades)
    cols = min(n, 6)
    total_w = cols * (CIRCLE_R * 2 + CIRCLE_GAP) - CIRCLE_GAP
    start_x = (w - total_w) // 2
    y = h - BOTTOM_H + 60
    positions = []
    for i, (name, bgr) in enumerate(shades):
        cx = start_x + i * (CIRCLE_R * 2 + CIRCLE_GAP) + CIRCLE_R
        positions.append((cx, y, name, bgr))
    return positions

def encode_frame(frame, quality=60):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

def decode_frame(frame_b64):
    frame_bytes = base64.b64decode(frame_b64)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
    status_msg = "Reconnecting..."
    status_color = (0, 0, 255)

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
            print(f"WebSocket failed: {e}")
        time.sleep(2)

def send_frame_to_stream(frame):
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
            haystyle_result = decode_frame(response.json()["frame"])
            hairstyle_result = haystyle_result
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

def draw_circle_filled(ui, cx, cy, r, bgr_color, selected=False):
    """Draw a filled circle swatch"""
    cv2.circle(ui, (cx, cy), r, bgr_color, -1)
    border_color = (255, 255, 255) if selected else (80, 80, 80)
    border_thickness = 3 if selected else 1
    cv2.circle(ui, (cx, cy), r, border_color, border_thickness)

def draw_ui(frame):
    global ui_state, selected_family
    h, w = frame.shape[:2]
    ui = frame.copy()

    # Top bar
    cv2.rectangle(ui, (0, 0), (w, 65), (15, 15, 15), -1)
    cv2.putText(ui, "ANGELS", (20, 46), cv2.FONT_HERSHEY_DUPLEX, 1.3, (200, 160, 255), 2)
    cv2.putText(ui, status_msg, (160, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 1)

    # Live indicator dot
    dot_color = (0, 255, 0) if ws_connected else (0, 0, 255)
    cv2.circle(ui, (w - 25, 32), 8, dot_color, -1)
    cv2.putText(ui, "LIVE" if ws_connected else "OFF", (w - 85, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)

    if processing:
        cv2.putText(ui, "Generating...", (w - 260, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # Bottom panel
    cv2.rectangle(ui, (0, h - BOTTOM_H), (w, h), (15, 15, 15), -1)
    cv2.line(ui, (0, h - BOTTOM_H), (w, h - BOTTOM_H), (50, 50, 50), 1)

    if ui_state == "families":
        # Label
        cv2.putText(ui, "HAIR COLOR", (20, h - BOTTOM_H + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Color family circles
        positions = build_family_positions(w, h)
        for i, (name, bgr) in enumerate(COLOR_FAMILIES):
            cx, cy = positions[i]
            is_selected = name == selected_family
            draw_circle_filled(ui, cx, cy, CIRCLE_R, bgr, is_selected)
            cv2.putText(ui, name[:6], (cx - 18, cy + CIRCLE_R + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

        # Original button
        orig_x = 20
        orig_y = h - BOTTOM_H + 110
        is_orig = current_color_name == "Original"
        cv2.rectangle(ui, (orig_x, orig_y), (orig_x + 90, orig_y + 30),
                      (200, 160, 255) if is_orig else (60, 60, 60), -1)
        cv2.rectangle(ui, (orig_x, orig_y), (orig_x + 90, orig_y + 30), (100, 100, 100), 1)
        cv2.putText(ui, "Original", (orig_x + 8, orig_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (20, 20, 20) if is_orig else (200, 200, 200), 1)

        # Hairstyle buttons
        hs_w, hs_h = 105, 32
        hs_cols = 4
        hs_x_start = w - hs_cols * (hs_w + 8) - 10
        cv2.putText(ui, "HAIRSTYLE TRY-ON", (hs_x_start, h - BOTTOM_H + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        for i, style in enumerate(HAIRSTYLES):
            col = i % hs_cols
            row = i // hs_cols
            bx = hs_x_start + col * (hs_w + 8)
            by = h - BOTTOM_H + 35 + row * (hs_h + 6)
            cv2.rectangle(ui, (bx, by), (bx + hs_w, by + hs_h), (50, 50, 50), -1)
            cv2.rectangle(ui, (bx, by), (bx + hs_w, by + hs_h), (80, 80, 80), 1)
            cv2.putText(ui, style[:13], (bx + 5, by + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (210, 210, 210), 1)

    elif ui_state == "shades":
        # Back button
        cv2.rectangle(ui, (10, h - BOTTOM_H + 8), (80, h - BOTTOM_H + 34), (50, 50, 50), -1)
        cv2.putText(ui, "< Back", (15, h - BOTTOM_H + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 160, 255), 1)

        cv2.putText(ui, f"{selected_family} shades", (95, h - BOTTOM_H + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Shade circles
        shade_positions = build_shade_positions(w, h, selected_family)
        for cx, cy, name, bgr in shade_positions:
            is_selected = bgr == current_color_bgr
            draw_circle_filled(ui, cx, cy, CIRCLE_R + 4, bgr, is_selected)
            # Two-line name
            words = name.split()
            if len(words) > 1:
                cv2.putText(ui, words[0], (cx - 20, cy + CIRCLE_R + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
                cv2.putText(ui, ' '.join(words[1:]), (cx - 20, cy + CIRCLE_R + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
            else:
                cv2.putText(ui, name, (cx - 20, cy + CIRCLE_R + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

    # Back button from result
    if mode == "result":
        cv2.rectangle(ui, (10, 75), (120, 115), (40, 40, 40), -1)
        cv2.rectangle(ui, (10, 75), (120, 115), (100, 100, 100), 1)
        cv2.putText(ui, "< BACK", (18, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 160, 255), 1)

    return ui

def mouse_click(event, x, y, flags, param):
    global current_color_bgr, current_color_name, processing
    global last_frame, mode, ui_state, selected_family

    if event != cv2.EVENT_LBUTTONDOWN or last_frame is None:
        return

    h, w = last_frame.shape[:2]

    # Back from result
    if mode == "result" and 10 <= x <= 120 and 75 <= y <= 115:
        mode = "live"
        return

    if ui_state == "families":
        # Original button
        orig_x, orig_y = 20, h - BOTTOM_H + 110
        if orig_x <= x <= orig_x + 90 and orig_y <= y <= orig_y + 30:
            current_color_bgr = None
            current_color_name = "Original"
            mode = "live"
            return

        # Color family circles
        positions = build_family_positions(w, h)
        for i, (name, bgr) in enumerate(COLOR_FAMILIES):
            cx, cy = positions[i]
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= CIRCLE_R + 5:
                selected_family = name
                ui_state = "shades"
                return

        # Hairstyle buttons
        if not processing:
            hs_w, hs_h = 105, 32
            hs_cols = 4
            hs_x_start = w - hs_cols * (hs_w + 8) - 10
            for i, style in enumerate(HAIRSTYLES):
                col = i % hs_cols
                row = i // hs_cols
                bx = hs_x_start + col * (hs_w + 8)
                by = h - BOTTOM_H + 35 + row * (hs_h + 6)
                if bx <= x <= bx + hs_w and by <= y <= by + hs_h:
                    processing = True
                    threading.Thread(target=request_hairstyle,
                                     args=(last_frame.copy(), style)).start()
                    return

    elif ui_state == "shades":
        # Back button
        if 10 <= x <= 80 and h - BOTTOM_H + 8 <= y <= h - BOTTOM_H + 34:
            ui_state = "families"
            return

        # Shade circles
        shade_positions = build_shade_positions(w, h, selected_family)
        for cx, cy, name, bgr in shade_positions:
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= CIRCLE_R + 9:
                current_color_bgr = bgr
                current_color_name = name
                mode = "live"
                return


# Start WebSocket thread
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
        send_frame_to_stream(frame)
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
    elif key == ord('b'):
        mode = "live"
        status_msg = "Live mode"
        status_color = (255, 255, 255)

cap.release()
cv2.destroyAllWindows()