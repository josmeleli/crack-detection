import numpy as np
import cv2 as cv

def detect_and_measure_circle(image_path: str):
    # Importando la imagen
    img = cv.imread(image_path)

    # Convirtiendo de espacio de color BGR a HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Definiendo límites inferior y superior para el color rojo
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Creando máscaras para ambos rangos de rojo y luego combinándolas
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)

    # Operación morfológica para reducir el ruido en la máscara
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

    # Filtrando el color rojo de la imagen original usando la máscara
    res = cv.bitwise_and(img, img, mask=mask)

    # Conversión a escala de grises para mejorar la detección de círculos
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    # Aplicando desenfoque gaussiano para reducir el ruido y mejorar la detección de círculos
    gray_blurred = cv.GaussianBlur(gray, (9, 9), 2, 2)

    # Aplicando detección de círculos de Hough
    circles = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, gray_blurred.shape[0]/8, param1=100, param2=30, minRadius=0, maxRadius=0)

    # Dibujando el círculo más grande y mostrando el diámetro si se detecta alguno
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        cv.circle(img, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 5)
        cv.circle(img, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 5)
        
        # Calculando el diámetro
        diameter = largest_circle[2] * 2

        return diameter
    else:
        return None

# Ejemplo de uso
image_path = 'C:/Users/USUARIO/Documents/Segmentation/66mm.jpg'
diameter = detect_and_measure_circle(image_path)
if diameter is not None:
    print(f"Diámetro del círculo detectado: {diameter} píxeles")
else:
    print("No se detectaron círculos.")