import cv2
import numpy as np
from math import sqrt
import mediapipe as mp

def inicializar_modelos():
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    modelos = {
        'hands': mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5),
        'face_mesh': mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5),
        'pose': mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5),
        'selfie_segmentation': mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    }

    return modelos

def remover_fundo(img, seg_model):
    seg_results = seg_model.process(img)
    condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(img.shape, dtype=np.uint8)
    bg_image[:] = (255, 255, 255)  # Fundo branco
    img_sem_fundo = np.where(condition, img, bg_image)
    return img_sem_fundo

def detectar_pessoa(img, modelos, width, height):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    hand_results = modelos['hands'].process(img)
    face_results = modelos['face_mesh'].process(img)
    pose_results = modelos['pose'].process(img)

    partes_identificadas = {f"{i}": [] for i in range(33)}
    esqueleto_img = img.copy()
    pontos_img = img.copy()
    medidas = []

    if pose_results.pose_landmarks:
        for id, lm in enumerate(pose_results.pose_landmarks.landmark):
            x, y = int(lm.x * width), int(lm.y * height)
            partes_identificadas[f"{id}"].append((x, y))
            cv2.circle(pontos_img, (x, y), 5, (0, 0, 255), cv2.FILLED)

        # Desenhar esqueleto e calcular medidas
        conexoes = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Cabeça
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Tronco e braços
            (11, 23), (12, 24), # Ombros aos quadris
            (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), # Quadris e pernas
            (27, 29), (29, 31), (28, 30), (30, 32) # Pernas inferiores
        ]
        for (i, j) in conexoes:
            ponto1 = pose_results.pose_landmarks.landmark[i]
            ponto2 = pose_results.pose_landmarks.landmark[j]
            x1, y1 = int(ponto1.x * width), int(ponto1.y * height)
            x2, y2 = int(ponto2.x * width), int(ponto2.y * height)
            distancia = int(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            medidas.append((i, j, distancia))
            cv2.line(esqueleto_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(esqueleto_img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(esqueleto_img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
            # Adiciona o texto com a medida na metade do segmento
            x_meio, y_meio = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(esqueleto_img, f'{distancia}px', (x_meio, y_meio), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return partes_identificadas, esqueleto_img, pontos_img, medidas

