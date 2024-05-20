import cv2
import mediapipe as mp
from cvzone.FaceDetectionModule import FaceDetector
import socket
import threading
from queue import Queue
import time

# Variáveis globais
video = cv2.VideoCapture(0)
detector = FaceDetector()
clients = []  # Lista para manter os clientes conectados

# Inicialize o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Fila para o último texto dito
last_said = Queue(maxsize=10)

def broadcast_message(message):
    for client in clients:
        try:
            client.send(message.encode('utf-8'))
        except:
            clients.remove(client)

def client_handler(client_socket):
    while True:
        try:
            message = last_said.get(block=True)  # Bloqueia até que haja algo para enviar
            broadcast_message(message)
        except:
            client_socket.close()
            break

def accept_connections(server):
    while True:
        client_sock, addr = server.accept()
        print(f"Conexão aceita de {addr}")
        clients.append(client_sock)
        client_thread = threading.Thread(target=client_handler, args=(client_sock,))
        client_thread.daemon = True
        client_thread.start()
        
def setup():
    # Configuração do servidor
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 8099))
    server.listen(5)
    print("Servidor escutando na porta 8099")

    accept_thread = threading.Thread(target=accept_connections, args=(server,))
    accept_thread.daemon = True
    accept_thread.start()        

def loop():
    while True:
        _, img = video.read()
        height, width, _ = img.shape
        
        # Detecção de rosto
        img, bboxes = detector.findFaces(img, draw=True)

        # Detecção de mãos
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Construir a mensagem no formato desejado
        timestamp = int(time.time() * 1000)
        message = f"inicio:\nframe:{timestamp};{width},{height}\n"
        
        if bboxes:
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox["bbox"]
                message += f"cabeca:{i+1};1;{x},{y};{x+w},{y+h}\n"
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])
                message += f"mao:{i+1};2;{int(x_min * width)},{int(y_min * height)};{int(x_max * width)},{int(y_max * height)}\n"
        
        message += "fim:"
        
        # Mostrar imagem com detecção
        cv2.imshow('Resultado', img)
        
        # Enviar informações sobre rosto e mãos
        broadcast_message(message)
        
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    setup()
    loop()

# Libere recursos
video.release()
cv2.destroyAllWindows()
hands.close()

