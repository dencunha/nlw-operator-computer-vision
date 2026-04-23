import cv2
import numpy as np
import base64

def draw_skeleton(frame, hand_landmarks):
    """Desenha o esqueleto da mão no frame, imitando o padrão do MediaPipe."""
    height, width, _ = frame.shape
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]
    for p1, p2 in connections:
        cv2.line(frame, points[p1], points[p2], (255, 255, 255), 2)
    for i, pt in enumerate(points):
        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        cv2.circle(frame, pt, 5, color, -1)

def decode_image(data_url):
    try:
        _, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return None

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"