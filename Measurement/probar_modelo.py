import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

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

# Ruta de la imagen de prueba
img_path = 'C:/Users/USUARIO/Desktop/Concrete-crack-detection/Images/_002.png'

# Cargar y procesar la imagen
img = load_img(img_path, target_size=(416, 416))  # Ajusta el tamaño si es necesario
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión para el batch
img_array = img_array / 255.0  # Normalizar la imagen si es necesario

# Realizar la predicción
prediction = model.predict(img_array)

# Si el modelo genera una máscara binaria, puedes visualizarla así:
predicted_mask = (prediction > 0.5).astype(np.uint8)

# Mostrar la máscara
plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

# Guardar la máscara como PNG
output_path = 'C:/Users/USUARIO/Desktop/docmodelskaggle/SemanticSegmentationTrainEvalmixup/imagenesconmascara/predicted_mask_kaggle_local.png'
plt.imsave(output_path, predicted_mask[0, :, :, 0], cmap='gray')

print(f'La máscara predicha ha sido guardada en {output_path}')
