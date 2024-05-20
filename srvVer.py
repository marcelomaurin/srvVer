
import cv2
import socket
import threading
from queue import Queue
import time
import subprocess
import sys

# Função para instalar bibliotecas ausentes
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verifica e instala as bibliotecas necessárias
try:
    import mediapipe as mp
except ImportError:
    install("mediapipe")
    import mediapipe as mp

try:
    from cvzone.FaceDetectionModule import FaceDetector
except ImportError:
    install("cvzone")
    from cvzone.FaceDetectionModule import FaceDetector


# Variáveis globais
video = cv2.VideoCapture(0)
detector = FaceDetector()
clients = []  # Lista para manter os clientes conectados

# Inicialize o MediaPipe Hands, FaceMesh e Pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
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

def identificar_cabeca(img, face_results, width, height):
    partes_identificadas = {
        "nariz": [],
        "olho_esquerdo": [],
        "olho_direito": [],
        "boca": [],
        "orelha_esquerda": [],
        "orelha_direita": [],
        "labio_superior": [],
        "labio_inferior": []
    }

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * width), int(lm.y * height)
                if id == 1:  # ID do nariz no FaceMesh
                    partes_identificadas["nariz"].append((x, y))
                    cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
                elif id == 33:  # ID do olho esquerdo no FaceMesh
                    partes_identificadas["olho_esquerdo"].append((x, y))
                    cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
                elif id == 263:  # ID do olho direito no FaceMesh
                    partes_identificadas["olho_direito"].append((x, y))
                    cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
                elif id in [13, 14]:  # IDs da boca no FaceMesh (13 = lábio inferior, 14 = lábio superior)
                    partes_identificadas["boca"].append((x, y))
                    cv2.circle(img, (x, y), 5, (0, 255, 255), cv2.FILLED)
                elif id == 127:  # ID da orelha esquerda no FaceMesh
                    partes_identificadas["orelha_esquerda"].append((x, y))
                    cv2.circle(img, (x, y), 5, (255, 255, 0), cv2.FILLED)
                elif id == 356:  # ID da orelha direita no FaceMesh
                    partes_identificadas["orelha_direita"].append((x, y))
                    cv2.circle(img, (x, y), 5, (255, 0, 255), cv2.FILLED)
                elif id in [61, 185, 39, 95, 88, 178, 191, 80, 81, 82]:  # IDs do lábio superior no FaceMesh
                    partes_identificadas["labio_superior"].append((x, y))
                    cv2.circle(img, (x, y), 2, (0, 128, 255), cv2.FILLED)
                elif id in [0, 17, 84, 91, 146, 61, 185, 195, 32, 193]:  # IDs do lábio inferior no FaceMesh
                    partes_identificadas["labio_inferior"].append((x, y))
                    cv2.circle(img, (x, y), 2, (128, 0, 255), cv2.FILLED)
            mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    return partes_identificadas

def identificar_maos(img, hand_results, width, height):
    partes_identificadas = {
        "maos": []
    }

    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            partes_identificadas["maos"].append((int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)))
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return partes_identificadas

def identificar_ombros(img, pose_results, width, height):
    partes_identificadas = {
        "ombro_esquerdo": [],
        "ombro_direito": []
    }

    if pose_results.pose_landmarks:
        for id, lm in enumerate(pose_results.pose_landmarks.landmark):
            x, y = int(lm.x * width), int(lm.y * height)
            if id == 11:  # ID do ombro esquerdo no Pose
                partes_identificadas["ombro_esquerdo"].append((x, y))
                cv2.circle(img, (x, y), 5, (255, 165, 0), cv2.FILLED)
            elif id == 12:  # ID do ombro direito no Pose
                partes_identificadas["ombro_direito"].append((x, y))
                cv2.circle(img, (x, y), 5, (0, 165, 255), cv2.FILLED)

    return partes_identificadas

def loop():
    while True:
        _, img = video.read()
        height, width, _ = img.shape
        
        # Detecção de rosto
        img, bboxes = detector.findFaces(img, draw=True)

        # Detecção de mãos e pose
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(img_rgb)
        face_results = face_mesh.process(img_rgb)
        pose_results = pose.process(img_rgb)
        
        # Identificar partes da cabeça
        partes_cabeca = identificar_cabeca(img, face_results, width, height)
        # Identificar mãos
        partes_maos = identificar_maos(img, hand_results, width, height)
        # Identificar ombros
        partes_ombros = identificar_ombros(img, pose_results, width, height)

        # Construir a mensagem no formato desejado
        timestamp = int(time.time() * 1000)
        message = f"inicio:\nframe:{timestamp};{width},{height}\n"
        
        if bboxes:
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox["bbox"]
                message += f"cabeca:{i+1};1;{x},{y};{x+w},{y+h}\n"
        
        for i, (x_min, y_min, x_max, y_max) in enumerate(partes_maos["maos"]):
            message += f"mao:{i+1};2;{x_min},{y_min};{x_max},{y_max}\n"

        for tipo, coords in partes_cabeca.items():
            for i, (x, y) in enumerate(coords):
                message += f"{tipo}:{i+1};3;{x},{y}\n"

        for tipo, coords in partes_ombros.items():
            for i, (x, y) in enumerate(coords):
                message += f"{tipo}:{i+1};4;{x},{y}\n"

        message += "fim:"
        
        # Mostrar imagem com detecção
        cv2.imshow('Resultado', img)
        
        # Enviar informações sobre rosto, mãos e ombros
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
face_mesh.close()
pose.close()
