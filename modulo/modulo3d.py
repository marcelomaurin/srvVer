import numpy as np
import open3d as o3d
import json

# Inicializa um dicionário global para armazenar os pontos 3D com rótulos
pontos_3d_dict = {}

def Inicializa_mapa3D():
    global pontos_3d_dict
    pontos_3d_dict = {}

def IncluiReferencia3D(pts1, pts2, baseline, fov, image_size, labels):
    global pontos_3d_dict
    
    # Calcula os parâmetros intrínsecos das câmeras
    focal_length = (image_size[0] / 2) / np.tan(np.radians(fov / 2))
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]])
    
    # Matriz de rotação (identidade para câmeras paralelas)
    R = np.eye(3)
    
    # Vetor de translação (distância entre as câmeras)
    T = np.array([baseline, 0, 0])
    
    # Converte os pontos para coordenadas homogêneas
    def to_homogeneous(points):
        return np.hstack([points, np.ones((points.shape[0], 1))])
    
    pts1_h = to_homogeneous(np.array(pts1))
    pts2_h = to_homogeneous(np.array(pts2))
    
    # Matriz de projeção para as câmeras
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, T.reshape(3, 1))))
    
    # Triangula os pontos
    def triangulate_point(p1, p2, P1, P2):
        A = np.array([
            p1[0] * P1[2, :] - P1[0, :],
            p1[1] * P1[2, :] - P1[1, :],
            p2[0] * P2[2, :] - P2[0, :],
            p2[1] * P2[2, :] - P2[1, :]
        ])
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        return X / X[3]
    
    points_3d = np.array([triangulate_point(p1, p2, P1, P2)[:3] for p1, p2 in zip(pts1_h, pts2_h)])
    
    # Adiciona os pontos e rótulos ao dicionário global
    for point, label in zip(points_3d, labels):
        pontos_3d_dict[label] = point.tolist()

def VisualizaMapa3d():
    global pontos_3d_dict

    # Cria uma nuvem de pontos em Open3D
    pcd = o3d.geometry.PointCloud()
    points = np.array(list(pontos_3d_dict.values()))
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Visualiza a nuvem de pontos
    o3d.visualization.draw_geometries([pcd])
    
    # Gera saída JSON
    output_json = json.dumps(pontos_3d_dict, indent=4)
    print(output_json)
    return output_json

