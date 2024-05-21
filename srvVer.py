import cv2
import socket
import threading
from queue import Queue
import time
import sys
from modulo import pessoa
from modulo import barcode
from modulo import formasbasicas

# Variáveis globais
video = cv2.VideoCapture(0)
clients = []  # Lista para manter os clientes conectados

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
    modelos = pessoa.inicializar_modelos()
    
    while True:
        _, img = video.read()
        height, width, _ = img.shape

        # Processar imagem para detecção de pessoa e remoção de fundo
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_results = pessoa.remover_fundo(img_rgb, modelos['selfie_segmentation'])
        img_sem_fundo = cv2.cvtColor(seg_results, cv2.COLOR_RGB2BGR)

        # Detecção de rosto, mãos e pose
        deteccoes, esqueleto_img, pontos_img, medidas = pessoa.detectar_pessoa(img_sem_fundo, modelos, width, height)

        # Ler código de barras
        barcode_data, barcode_rect = barcode.read_barcode(img)

        # Detectar formas básicas
        formas = formasbasicas.FormasBasicas(img)

        # Construir a mensagem no formato desejado
        timestamp = int(time.time() * 1000)
        message = f"inicio:\nframe:{timestamp};{width},{height}\n"
        
        for tipo, coords in deteccoes.items():
            for i, (x, y) in enumerate(coords):
                message += f"{tipo}:{i+1};4;{x},{y}\n"

        for i, j, distancia in medidas:
            message += f"medida:{i}-{j};{distancia}\n"

        if barcode_data and barcode_rect:
            x, y, w, h = barcode_rect
            message += f"barcode:{barcode_data};{x},{y};{x+w},{y+h}\n"
            cv2.rectangle(esqueleto_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(esqueleto_img, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for forma in formas:
            shape_type = forma["shape"]
            coords = forma["coordinates"]
            for coord in coords:
                x, y = coord
                message += f"{shape_type}:1;5;{x},{y}\n"
            if len(coords) >= 4:
                x, y = coords[0]
                w = coords[2][0] - x
                h = coords[2][1] - y
                cv2.rectangle(esqueleto_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(esqueleto_img, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        message += "fim:"
        
        # Mostrar imagens
        cv2.imshow('Imagem Original', img)
        cv2.imshow('Esqueleto', esqueleto_img)
        cv2.imshow('Pontos Lidos', pontos_img)
        
        # Enviar informações sobre rosto, mãos e pose
        broadcast_message(message)
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    setup()
    loop()

# Libere recursos
video.release()
cv2.destroyAllWindows()

