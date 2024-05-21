import cv2

def FormasBasicas(image):
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desfoque para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas na imagem
    edged = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lista para armazenar as formas detectadas
    shapes = []
    
    # Iterar sobre os contornos e identificar formas
    for contour in contours:
        # Aproximação do contorno a uma forma poligonal
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        shape_type = ""
        coordinates = approx.reshape(-1, 2).tolist()
        
        # Classificar a forma com base no número de vértices
        if len(approx) == 3:
            shape_type = "Triângulo"
        elif len(approx) == 4:
            # Verificar se é um quadrado ou retângulo
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape_type = "Quadrado" if 0.95 <= aspect_ratio <= 1.05 else "Retângulo"
        elif len(approx) > 4:
            shape_type = "Círculo"
        
        # Adicionar a forma à lista de formas detectadas
        shapes.append({"shape": shape_type, "coordinates": coordinates})
    
    return shapes

