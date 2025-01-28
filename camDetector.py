import cv2
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilizando dispositivo: {device}")

model = YOLO('modelos/best.pt')

classNames = ['Glasses', 'no-glasses']

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Erro ao acessar a webcam. Verifique as configurações.")
    exit()

print("Transmissão ao vivo iniciada com detecção de objetos. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível obter o frame da webcam.")
        break

    results = model.predict(source=frame, device=device, verbose=False)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = round(float(result.conf[0]), 2)
        cls = int(result.cls[0])
        currentClass = classNames[cls]
        color = (0, 255, 0) if 'NO-' not in currentClass else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{currentClass} {conf}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Transmissão ao Vivo com Detecção de Objetos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
