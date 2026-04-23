import json, time
from fasthtml.common import *
import cv2
import numpy as np
import base64
from core.processor import GestureProcessor

app, rt = fast_app(exts=['ws'])
processor = GestureProcessor()
last_time = time.time()

def decode_image(data_url):
    try:
        _, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return None

def encode_image(img, quality=70):
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@rt('/assets/{fname:path}')
def serve_assets(fname: str):
    return FileResponse(f'assets/{fname}')

@rt("/")
def get():
    return Title("Rocket Vision"), \
           Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"), \
           Link(rel="stylesheet", href="/assets/style.css"), Main(
        
        Div(
            # CORRIGIDO: Rocket Vision
            H1("Rocket Vision"),
            P("Intelligent Hand Gesture Recognition System"),
            cls="header"
        ),
        
        Div(
            # Coluna Esquerda: Webcam e Controles
            Div(
                Div(
                    Video(id="video", autoplay=True, playsinline=True, style="display:none;"),
                    Canvas(id="canvas"),
                    cls="video-card"
                ),
                # Painel 3: Controles (Movido para baixo da webcam)
                Div(
                    Div("SETTINGS", cls="card-title"),
                    Div(
                        # Grupo do Slider (Ocupará a maior parte)
                        Div(
                            Div(
                                Label("Image Quality", fr="quality-slider"),
                                Span("0.6", id="quality-value"),
                                cls="slider-header"
                            ),
                            Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.1", value="0.6"),
                            cls="control-group quality-group"
                        ),
                        # Grupo do Checkbox (Ficará no canto direito, alinhado ao centro)
                        Div(
                            Label("Show Landmarks", fr="show-landmarks"),
                            Input(type="checkbox", id="show-landmarks", checked=True),
                            cls="control-group checkbox-group"
                        ),
                        cls="settings-layout"
                    ),
                    cls="data-card settings-card"
                ),
            ),
            
            # Coluna Direita: Painéis
            Div(
                # Painel 1: Dados
                Div(
                    Div(
                        Span("LIVE FEED DATA"),
                        Span("0 FPS", id="fps-counter", cls="fps-badge"),
                        cls="card-title", style="display: flex; justify-content: space-between; align-items: center;"
                    ),
                    Div(id="gesture-container"),
                    cls="data-card"
                ),
                # Painel 2: Imagem
                Div(
                    Div("DETECTED GESTURE", cls="card-title"),
                    Img(id="gesture-image"),
                    cls="data-card", style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px;"
                ),
                cls="side-panel"
            ),
            cls="main-content"
        ),
        # Truque do cache ativado (v=4)
        Script(src="/assets/script.js?v=4"),
    )

@app.ws("/ws")
async def ws(image: str, show_landmarks: bool, quality: float, send):
    global last_time
    current_time = time.time()
    fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
    last_time = current_time

    img = decode_image(image)
    if img is not None:
        cv_quality = int(quality * 100)
        processed_image, labels, matching_image = processor.process_frame(img, show_landmarks=show_landmarks)
        
        await send(json.dumps({
            "image": encode_image(processed_image, quality=cv_quality),
            "labels": labels,
            "matching_image": matching_image,
            "fps": round(fps, 1)
        }))

if __name__ == "__main__":
    serve()