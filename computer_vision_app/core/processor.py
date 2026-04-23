import cv2
import mediapipe as mp
import numpy as np
from core.models import get_mediapipe_options, load_custom_models
from core.utils import draw_skeleton

class GestureProcessor:
    def __init__(self):
        self.clf, self.label_encoder = load_custom_models()
        self.options = get_mediapipe_options()
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)

    def process_frame(self, frame, show_landmarks=True):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        labels = []
        # Nova variável para guardar os nomes puros dos gestos detectados
        detected_gestures = [] 

        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                if show_landmarks:
                    draw_skeleton(frame, hand_landmarks)

                google_category = "None"
                google_score = 0.0
                if recognition_result.gestures and len(recognition_result.gestures) > i:
                    google_category = recognition_result.gestures[i][0].category_name
                    google_score = recognition_result.gestures[i][0].score

                raw_handedness = recognition_result.handedness[i][0].category_name
                hand_label = "Right" if raw_handedness == "Left" else "Left"
                handedness_val = 0 if hand_label == 'Left' else 1
                
                landmarks_array = [handedness_val]
                for lm in hand_landmarks:
                    landmarks_array.extend([lm.x, lm.y, lm.z])
                
                features = np.array(landmarks_array).reshape(1, -1)
                prediction_idx = self.clf.predict(features)[0]
                prediction_prob = np.max(self.clf.predict_proba(features))
                custom_gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                if google_category != "None" and google_category != "":
                    final_gesture = google_category.lower()
                    final_prob = google_score
                    detected_gestures.append(final_gesture)
                else:
                    final_gesture = custom_gesture_name.lower()
                    final_prob = prediction_prob
                    detected_gestures.append(final_gesture)
                
                # Mandamos os dados limpos e separados para o JS montar o design!
                labels.append({
                    "hand": hand_label,
                    "gesture": final_gesture,
                    "probability": float(final_prob)
                })

# --- A MÁGICA DOS 2 GESTOS IGUAIS ---
        matching_image = None
        
        # O NOSSO ESPIÃO: Vai imprimir no terminal do Linux o que as mãos estão fazendo
        if len(detected_gestures) == 2:
            print(f"🕵️ Gestos lidos: Mão 1 = '{detected_gestures[0]}' | Mão 2 = '{detected_gestures[1]}'")

        if len(detected_gestures) == 2 and detected_gestures[0] == detected_gestures[1]:
            gesture = detected_gestures[0]
            
            # O "Tradutor" com os nomes EXATOS dos seus arquivos
            mapping = {
                "thumb_up": "joinha.png",
                "thumbs_up": "joinha.png", 
                "victory": "paz.png",
                "paz": "paz.png",
                "open_palm": "ola.png",
                "hi": "ola.png",
                "closed_fist": "rock.png",
                "rock": "rock.png",
                "spock": "spock.png",
                "hearts": "coracao.png",
                "coracao": "coracao.png",
                "hangloose": "hangloose.png"
            }
            matching_image = mapping.get(gesture)
            print(f"🎯 MATCH! Imagem escolhida: {matching_image}")

        return frame, labels, matching_image

    def close(self):
        if self.recognizer:
            self.recognizer.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()