from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2 as cv
from skimage import morphology
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Define la capa personalizada FixedDropout
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape, seed, **kwargs)

    def get_config(self):
        config = super(FixedDropout, self).get_config()
        return config

# Especifica la ruta de tu modelo .h5
model_path = 'C:/Users/USUARIO/Desktop/docmodelskaggle/SemanticSegmentationTrainEvalmixup/model_effnetB1_512pix_fold3_full_84283.h5'

# Cargar el modelo con la capa personalizada
model = load_model(model_path, custom_objects={'FixedDropout': FixedDropout}, compile=False)

# Definir la función de medición de ancho de grieta
def crack_width_measure(binary_image):
    """
    Medir el ancho de la grieta a lo largo de su longitud en una imagen binaria.
    
    :param binary_image: Una matriz NumPy 2D que representa la imagen binaria (donde la grieta se representa como píxeles blancos).
    :return: Una tupla que contiene una matriz de anchos de grieta y el ancho máximo de la grieta.
    """
    # Asegurando que la imagen utilizada sea una imagen binaria
    binary_image = binary_image > 0
    
    # Aplicando esqueletonización
    skeleton = morphology.skeletonize(binary_image)
    
    # Aplicando la transformación de distancia
    distance_transform = distance_transform_edt(binary_image)
    
    # Midiendo los anchos de la grieta a lo largo del esqueleto
    crack_widths = distance_transform[skeleton] * 2
    
    # Identificando el ancho máximo de la grieta en la imagen
    max_crack_width = np.max(crack_widths) if crack_widths.size > 0 else 0
    
    return crack_widths, max_crack_width

# Definir la aplicación FastAPI
app = FastAPI()

@app.post("/ancho-maximo-grieta/")
async def measure_crack_width(file: UploadFile = File(...)):
    # Leer el archivo de imagen
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocesar la imagen
    img = image.resize((416, 416))  # Redimensionar si es necesario
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para el batch
    img_array = img_array / 255.0  # Normalizar la imagen
    
    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_mask = (prediction > 0.5).astype(np.uint8)[0, :, :, 0]
    
    # Guardar la máscara como PNG
    output_path = 'predicted_mask.png'
    plt.imsave(output_path, predicted_mask, cmap='gray')

    # Leer la imagen binaria como bytes
    buffered = io.BytesIO()
    Image.fromarray(predicted_mask * 255).save(buffered, format="PNG")
    binary_image_bytes = buffered.getvalue()
    
    # Medir el ancho de la grieta
    crack_widths, max_crack_width = crack_width_measure(predicted_mask)

    return JSONResponse(content={"max_crack_width": max_crack_width})

# Cargar y procesar la imagen
img_path = 'c:/Users/USUARIO/Documents/Segmentation/ejemplopostman.jpg'

# Procesar y medir el ancho de grieta directamente en Python
def process_and_measure_crack(image_path):
    # Cargar la imagen
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    
    # Convertir imagen a array de numpy
    image_array = np.array(image)
    
    # Aplicar umbral para crear una imagen binaria
    threshold = 128
    binary_image_array = (image_array > threshold).astype(np.uint8)
    
    # Medir el ancho de la grieta
    crack_widths, max_crack_width = crack_width_measure(binary_image_array)
    
    return crack_widths, max_crack_width



# Procesar y medir el ancho de grieta directamente en Python
def process_and_measure_crack(image_path):
    # Cargar la imagen
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    
    # Convertir imagen a array de numpy
    image_array = np.array(image)
    
    # Aplicar umbral para crear una imagen binaria
    threshold = 128
    binary_image_array = (image_array > threshold).astype(np.uint8)
    
    # Medir el ancho de la grieta
    crack_widths, max_crack_width = crack_width_measure(binary_image_array)
    
    return crack_widths, max_crack_width

# Function to detect circles
def detect_and_measure_circle(image: np.ndarray):
    # Convirtiendo de espacio de color BGR a HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

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
    res = cv.bitwise_and(image, image, mask=mask)

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
        cv.circle(image, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 5)
        cv.circle(image, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 5)
        
        # Calculando el diámetro
        diameter = largest_circle[2] * 2
        
        return diameter
    else:
        return None

@app.post("/detectar-circulos/")
async def detect_circles_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv.cvtColor(np.array(Image.open(io.BytesIO(contents))), cv.COLOR_RGB2BGR)

    diameter = detect_and_measure_circle(image)

    if diameter is not None:
        return JSONResponse(content={"message": f"El diámetro del círculo es {diameter} píxeles"})
    else:
        return JSONResponse(content={"message": "No se detectaron círculos."})



# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
