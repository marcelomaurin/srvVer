import cv2
import socket
import threading
from queue import Queue
import time
import sys
from modulo import pessoa
from modulo import barcode
from modulo import formasbasicas
from modulo import modulo3d

# Variáveis globais
video1 = cv2.VideoCapture(0)
video2 = None
clients = []  # Lista para manter os clientes conectados
visao_stereo = False  # Definir para True para ativar a visão estéreo

# Parâmetros da visão estéreo
baseline = 0.1  # Distância entre as câmeras em metros
fov = 90  # Campo de visão da câmera em graus
image_size = (640, 480)  # Tamanho da imagem (largura, altura)

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
    global video2, visao_stereo
    
    # Configuração do servidor
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 8099))
    server.listen(5)
    print("Servidor escutando na porta 8099")

    if visao_stereo:
        video2 = cv2.VideoCapture(1)  # Inicializa a segunda câmera
        modulo3d.Inicializa_mapa3D()

    accept_thread = threading.Thread(target=accept_connections, args=(server,))
    accept_thread.daemon = True
    accept_thread.start()

def process_frame(img, modelos, width, height):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_results = pessoa.remover_fundo(img_rgb, modelos['selfie_segmentation'])
    img_sem_fundo = cv2.cvtColor(seg_results, cv2.COLOR_RGB2BGR)

    deteccoes, esqueleto_img, pontos_img, medidas = pessoa.detectar_pessoa(img_sem_fundo, modelos, width, height)

    barcode_data, barcode_rect = barcode.read_barcode(img)
    formas = formasbasicas.FormasBasicas(img)

    return deteccoes, esqueleto_img, pontos_img, medidas, barcode_data, barcode_rect, formas

def loop():
    global visao_stereo
    
    modelos = pessoa.inicializar_modelos()
    
    while True:
        _, img1 = video1.read()
        height, width, _ = img1.shape

        results1 = process_frame(img1, modelos, width, height)
        
        if visao_stereo:
            _, img2 = video2.read()
            results2 = process_frame(img2, modelos, width, height)

        # Construir a mensagem no formato desejado
        timestamp = int(time.time() * 1000)
        message = f"inicio:\nframe:{timestamp};{width},{height}\n"
        
        for tipo, coords in results1[0].items():
            for i, (x, y) in enumerate(coords):
                message += f"{tipo}:{i+1};4;{x},{y}\n"

        for i, j, distancia in results1[3]:
            message += f"medida:{i}-{j};{distancia}\n"

        if results1[4] and results1[5]:
            x, y, w, h = results1[5]
            message += f"barcode:{results1[4]};{x},{y};{x+w},{y+h}\n"
            cv2.rectangle(results1[1], (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(results1[1], results1[4], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for forma in results1[6]:
            shape_type = forma["shape"]
            coords = forma["coordinates"]
            for coord in coords:
                x, y = coord
                message += f"{shape_type}:1;5;{x},{y}\n"
            if len(coords) >= 4:
                x, y = coords[0]
                w = coords[2][0] - x
                h = coords[2][1] - y
                cv2.rectangle(results1[1], (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(results1[1], shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if visao_stereo:
            # Processar a segunda câmera e incluir pontos 3D
            pts1 = [item for sublist in results1[0].values() for item in sublist]
            pts2 = [item for sublist in results2[0].values() for item in sublist]
            labels = [f"{tipo}:{i+1}" for tipo, coords in results1[0].items() for i in range(len(coords))]
            
            modulo3d.IncluiReferencia3D(pts1, pts2, baseline, fov, image_size, labels)

        message += "fim:"
        
        # Mostrar imagens
        cv2.imshow('Imagem Original', img1)
        cv2.imshow('Esqueleto', results1[1])
        cv2.imshow('Pontos Lidos', results1[2])
        
        # Enviar informações sobre rosto, mãos e pose
        broadcast_message(message)
        
        if cv2.waitKey(1) == 27:
            break

    if visao_stereo:
        # Visualizar o mapa 3D ao final
        modulo3d.VisualizaMapa3d()

if __name__ == "__main__":
    setup()
    loop()

# Libere recursos
video1.release()
if visao_stereo:
    video2.release()
cv2.destroyAllWindows()

