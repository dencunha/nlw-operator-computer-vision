import cv2
import mediapipe as mp
import os
import csv
import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminho para o modelo base
MODEL_PATH = "gesture_recognizer.task"

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
    parser = argparse.ArgumentParser(description='Coleta de hand landmarks para dataset.')
    parser.add_argument('--label', type=str, required=True, help='Label do gesto que será coletado')
    parser.add_argument('--output', type=str, default='hand_landmarks_data.csv', help='Arquivo CSV de saída')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O arquivo {MODEL_PATH} não foi encontrado na pasta atual.")
        return

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    file_exists = os.path.isfile(args.output)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Coletando dados para a label: {args.label}")
    print("Comandos:")
    print("  's' - Salva UMA captura dos landmarks atuais")
    print("  'r' - Inicia/Para gravação CONTÍNUA")
    print("  'q' - Sair")

    recording = False

    with GestureRecognizer.create_from_options(options) as recognizer:
        with open(args.output, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            
            if not file_exists:
                header = ['label', 'handedness']
                for i in range(21):
                    header.extend([f'x{i}', f'y{i}', f'z{i}'])
                csv_writer.writerow(header)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                
                recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                current_landmarks = None
                handedness = None
                if recognition_result.hand_landmarks:
                    current_landmarks = recognition_result.hand_landmarks[0]
                    raw_handedness = recognition_result.handedness[0][0].category_name
                    
                    # Correção do Espelho apenas para salvar no CSV corretamente
                    handedness = "Right" if raw_handedness == "Left" else "Left"
                    
                    # Usa a nossa função segura em vez da do Google!
                    desenhar_esqueleto(frame, current_landmarks)

                if recording and current_landmarks:
                    row = [args.label, handedness]
                    for lm in current_landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    csv_writer.writerow(row)

                status_color = (0, 255, 0) if recording else (255, 255, 255)
                status_text = "GRAVANDO" if recording else "STANDBY"
                cv2.putText(frame, f"Label: {args.label} [{status_text}]", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                if current_landmarks:
                    cv2.putText(frame, f"Mao detectada: {handedness}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Mao nao detectada", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow('Coleta de Dados - Landmark Recorder', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if current_landmarks:
                        row = [args.label, handedness]
                        for lm in current_landmarks:
                            row.extend([lm.x, lm.y, lm.z])
                        csv_writer.writerow(row)
                        f.flush()
                        print(f"Frame único salvo para label '{args.label}'")
                        cv2.rectangle(frame, (0,0), (640,480), (0,255,0), 10)
                        cv2.imshow('Coleta de Dados - Landmark Recorder', frame)
                        cv2.waitKey(50)
                elif key == ord('r'):
                    recording = not recording
                    print(f"Gravação contínua: {'INICIADA' if recording else 'PARADA'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()