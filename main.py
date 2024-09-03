import tempfile
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from io import BytesIO
import base64

# Define una clase personalizada para DepthwiseConv2D
class CustomDepthwiseConv2D(BaseDepthwiseConv2D):
    def __init__(self, **kwargs):
        # Elimina el argumento 'groups' ya que no es compatible
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Define una capa personalizada si es necesario
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape, seed, **kwargs)

    def get_config(self):
        config = super(FixedDropout, self).get_config()
        return config

# Inicializa la aplicación FastAPI
app = FastAPI()

# Ruta al archivo del modelo
model_path = 'models/model_effnetB1_512pix_fold3_full_84283.h5'

# Cargar el modelo con la capa personalizada
try:
    model = load_model(model_path, custom_objects={'FixedDropout': FixedDropout, 'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
    print("Modelo de segmentación cargado con éxito")
except Exception as e:
    print(f"Error al cargar el modelo de segmentación: {e}")

# Cargar el modelo de clasificación
classification_model_path = 'models/CNN_surface_crack_detection.h5'
try:
    classification_model = tf.keras.models.load_model(classification_model_path)
    print("Modelo de clasificación cargado con éxito")
except Exception as e:
    print(f"Error al cargar el modelo de clasificación: {e}")

# Función para medir el ancho de las grietas
def crack_width_measure(binary_image, display_results=True):
    binary_image = binary_image > 0
    
    # Aplicando skeletonisation
    skeleton = morphology.skeletonize(binary_image)
    
    # Aplicando distance transform
    distance_transform = distance_transform_edt(binary_image)
    
    # Midiendo los anchos de grietas a lo largo del esqueleto
    crack_widths = distance_transform[skeleton] * 2
    
    # Identificando el ancho máximo de la grieta en la imagen
    max_crack_width = np.max(crack_widths) if crack_widths.size > 0 else 0
    
    if display_results:
        # Mostrar los anchos de grietas a lo largo del esqueleto
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_image, cmap='gray')
        plt.scatter(*np.where(skeleton)[::-1], c=crack_widths, cmap='jet', s=10, label='Crack Widths')
        plt.colorbar(label='Ancho Máximo en (pixels)')
        plt.title('Grieta con Ancho Máximo')
        plt.axis('off')
        
        # Guardar la imagen en memoria en lugar de en el disco
        buf = BytesIO()
        plt.savefig(buf, format='jpg', dpi=300)
        buf.seek(0)
        plt.close()
        
        # Convertir la imagen a base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return crack_widths, max_crack_width, image_base64
    
    return crack_widths, max_crack_width, None

# Función para convertir la imagen a binaria y medir el ancho de la grieta
def process_and_measure_crack(binary_image_array):
    crack_widths, max_crack_width, image_base64 = crack_width_measure(binary_image_array)
    return crack_widths, max_crack_width, image_base64

def load_and_preprocess_image(img_path):
    img = Image.open(img_path).resize((120, 120))  # Usar PIL para cargar y redimensionar
    img_array = np.array(img, dtype=np.float32)  # Asegúrate de que la imagen esté en float32
    img_array /= 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Endpoint POST para procesar la imagen
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Crear un archivo temporal para usar con load_img
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(contents)
        temp_image_path = temp_image.name

    # Usar PIL para cargar la imagen desde el archivo temporal
    image = Image.open(temp_image_path).resize((416, 416))

    # Preprocesar la imagen para el modelo
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar la imagen

    # Realizar la predicción de segmentación
    prediction = model.predict(img_array)
    predicted_mask = (prediction > 0.5).astype(np.uint8)

    # Convertir la máscara predicha a una imagen binaria
    binary_image_array = predicted_mask[0, :, :, 0]

    # Procesar y medir el ancho de la grieta
    crack_widths, max_crack_width, highlighted_image_base64 = crack_width_measure(binary_image_array)

    # Devolver los resultados
    return JSONResponse(content={"max_crack_width": max_crack_width, "max_width_image": highlighted_image_base64})

# Función para detectar y medir círculos
def detect_and_measure_circle(image: np.ndarray):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

    res = cv.bitwise_and(image, image, mask=mask)

    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    gray_blurred = cv.GaussianBlur(gray, (9, 9), 2, 2)

    circles = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, gray_blurred.shape[0] / 8, param1=100, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        cv.circle(image, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 5)
        cv.circle(image, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 5)
        
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

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(contents)
        temp_image_path = temp_image.name

    img_array = load_and_preprocess_image(temp_image_path)

    prediction = classification_model.predict(img_array)

    threshold = 0.5

    if prediction[0] > threshold:
        return JSONResponse(content={"prediction": "Crack Detected"})
    else:
        return JSONResponse(content={"prediction": "No Crack Detected"})

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
