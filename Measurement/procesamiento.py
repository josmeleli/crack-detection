import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from PIL import Image
import os

# Definir la capa personalizada FixedDropout
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

# Ruta de la imagen de prueba
img_path = 'c:/Users/USUARIO/Documents/Segmentation/ejemplopostman.jpg'

# Cargar y procesar la imagen
img = load_img(img_path, target_size=(416, 416))  # Ajusta el tamaño si es necesario
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión para el batch
img_array = img_array / 255.0  # Normalizar la imagen si es necesario

# Realizar la predicción
prediction = model.predict(img_array)

# Si el modelo genera una máscara binaria, puedes visualizarla así:
predicted_mask = (prediction > 0.5).astype(np.uint8)

# Guardar la máscara predicha como PNG
output_path = 'C:/Users/USUARIO/Desktop/docmodelskaggle/SemanticSegmentationTrainEvalmixup/imagenesconmascara/predicted_mask_h5_3.png'
plt.imsave(output_path, predicted_mask[0, :, :, 0], cmap='gray')

print(f'La máscara predicha ha sido guardada en {output_path}')

#-----------CREATING A CRACK WIDTH MEASUREMENT FUNCTION------------------------

def crack_width_measure(binary_image, display_results=True):
    """
    Measure the crack widths along its length and identify the maximum crack width in a binary image.
    
    :param binary_image: A 2D NumPy array representing the binary image (where the crack is represented as white pixels).
    :param display_results: If True, display the results.
    :return: A tuple containing an array of crack widths and the maximum crack width.
    """
    # Ensuring the image being used is a binary image
    binary_image = binary_image > 0
    
    # Applying skeletonisation
    skeleton = morphology.skeletonize(binary_image)
    
    # Applying distance transform
    distance_transform = distance_transform_edt(binary_image)
    
    # Measuring crack widths along the skeleton
    crack_widths = distance_transform[skeleton] * 2
    
    # Identifying maximum crack width in the image
    max_crack_width = np.max(crack_widths) if crack_widths.size > 0 else 0
    
    if display_results:
        # Display the original binary image
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image')
        plt.axis('off')
        plt.show()
        
        # Display the skeleton
        plt.figure(figsize=(6, 6))
        plt.imshow(skeleton, cmap='gray')
        plt.title('Skeleton')
        plt.axis('off')
        plt.show()
        
        # Display the distance transform
        plt.figure(figsize=(6, 6))
        plt.imshow(distance_transform, cmap='gray')
        plt.title('Distance Transform')
        plt.axis('off')
        plt.show()
        
        # Display the crack widths along the skeleton
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_image, cmap='gray')
        plt.scatter(*np.where(skeleton)[::-1], c=crack_widths, cmap='jet', s=10, label='Crack Widths')
        plt.colorbar(label='Crack Width (pixels)')
        plt.title('Detected Crack')
        plt.axis('off')
        
        # Print value of max crack width
        print(f"Maximum Crack Width: {max_crack_width} pixels")
        
        text_x = 10  # 10 pixels from the left
        text_y = 20  # 20 pixels from the top
        
        plt.text(text_x, text_y, f"Max Width: {max_crack_width:.2f} pixels", color='white')
        
        plt.savefig("crackwidth.jpg", dpi=300)
        plt.show()
        
    return crack_widths, max_crack_width

#CREATING A FUNCTION TO CONVERT IMAGE TO BINARY AND MEASURE CRACK WIDTH-------------------

def process_and_measure_crack(image_path, display_results=True):
    """
    Convert an image to binary format, save the binary image, and analyze the crack widths.
    
    :param image_path: Path to the input image file.
    :param display_results: If True, display the results.
    :return: A tuple containing an array of crack widths and the maximum crack width.
    """
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Apply threshold to create a binary image
    threshold = 128
    binary_image_array = (image_array > threshold).astype(np.uint8)
    
    # Convert binary array back to image
    binary_image = Image.fromarray(binary_image_array * 255)  # Scale values to 0 or 255
    
    # Save the binary image
    binary_output_path = image_path.replace('.png', '_binary.png')
    binary_image.save(binary_output_path)
    print(f"Binary image saved to: {binary_output_path}")
    
    # Measure crack width
    crack_widths, max_crack_width = crack_width_measure(binary_image_array, display_results)
    
    return crack_widths, max_crack_width

# USING THE FUNCTION
# Specify the path to the input image
input_image_path = output_path

# Process and measure crack width
crack_widths, max_crack_width = process_and_measure_crack(input_image_path)

print(f"Maximum crack width: {max_crack_width} pixels")
