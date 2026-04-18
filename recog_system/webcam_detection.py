import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminhos para os modelos
MP_MODEL_PATH = "gesture_recognizer.task"
CUSTOM_MODEL_PATH = "gesture_model.joblib"
ENCODER_PATH = "label_encoder.joblib"

# -------------------------------------------------------------------
# FUNÇÃO SÊNIOR: Bypass no bug de versão do MediaPipe!
# -------------------------------------------------------------------
def desenhar_esqueleto(frame, hand_landmarks):
    height, width, _ = frame.shape
    pontos = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
    conexoes = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]
    for p1, p2 in conexoes:
        cv2.line(frame, pontos[p1], pontos[p2], (255, 255, 255), 2)
    for i, pt in enumerate(pontos):
        cor = (0, 0, 255) if i == 0 else (0, 255, 0)
        cv2.circle(frame, pt, 5, cor, -1)
# -------------------------------------------------------------------

def main():
    if not all(os.path.exists(p) for p in [MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH]):
        print("Erro: Um ou mais arquivos de modelo não foram encontrados.")
        return

    print("--- Carregando modelo Híbrido ---")
    clf = joblib.load(CUSTOM_MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    print("\nIniciando sistema HÍBRIDO... Pressione 'q' para sair.")

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if recognition_result.hand_landmarks:
                for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                    
                    desenhar_esqueleto(frame, hand_landmarks)

                    # --- 1. PERGUNTA PARA O GOOGLE ---
                    google_category = "None"
                    google_score = 0.0
                    if recognition_result.gestures and len(recognition_result.gestures) > i:
                        google_category = recognition_result.gestures[i][0].category_name
                        google_score = recognition_result.gestures[i][0].score

                    # --- 2. PERGUNTA PARA A SUA IA ---
                    raw_handedness = recognition_result.handedness[i][0].category_name
                    hand_label = "Right" if raw_handedness == "Left" else "Left"
                    handedness_val = 0 if hand_label == 'Left' else 1
                    
                    landmarks_array = [handedness_val]
                    for lm in hand_landmarks:
                        landmarks_array.extend([lm.x, lm.y, lm.z])
                    
                    features = np.array(landmarks_array).reshape(1, -1)
                    prediction_idx = clf.predict(features)[0]
                    prediction_prob = np.max(clf.predict_proba(features))
                    custom_gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

                    # --- 3. LÓGICA DE DECISÃO SÊNIOR (O ÁRBITRO) ---
                    # Se o Google souber o que é (não for "None"), usamos ele (Azul)
                    # Se o Google não fizer ideia, usamos o seu modelo customizado (Verde)
                    if google_category != "None" and google_category != "":
                        display_text = f"Google {hand_label}: {google_category} ({google_score:.2f})"
                        cor_texto = (255, 150, 0) # Azul claro
                    else:
                        display_text = f"Minha IA {hand_label}: {custom_gesture_name} ({prediction_prob:.2f})"
                        cor_texto = (0, 255, 0) # Verde
                    
                    cv2.putText(frame, display_text, (20, 50 + (i * 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_texto, 2)

            cv2.imshow('Sistema Hibrido de Gestos', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()