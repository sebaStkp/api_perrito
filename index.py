from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

# Inicializar la aplicación FastAPI
app = FastAPI()

# Configurar CORS para permitir solicitudes desde cualquier origen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir la ruta del modelo
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'TallerPerritos.keras')

# Cargar el modelo al iniciar la aplicación
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    """
    Carga y preprocesa la imagen para que sea compatible con el modelo TensorFlow.
    """
    img = Image.open(image_path).convert('RGB')  # Convertir a RGB
    img = img.resize((224, 224))  # Redimensionar a 224x224
    img_array = np.array(img)  # Convertir a un arreglo numpy
    img_array = preprocess_input(img_array)  # Preprocesar la imagen
    return np.expand_dims(img_array, axis=0)  # Agregar una dimensión adicional para lotes

@app.post("/classify_pet")
async def classify_pet(file: UploadFile = File(...)):
    """
    Endpoint para clasificar si la imagen subida es de un perro o no.
    """
    try:
        # Verificar si el modelo está cargado
        if model is None:
            return {"error": "El modelo no se cargó correctamente."}

        # Guardar la imagen temporalmente
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Preprocesar la imagen
        processed_image = load_and_preprocess_image(temp_image_path)

        # Realizar la predicción
        prediction = model.predict(processed_image)
        confidence = prediction[0][0]

        # Eliminar la imagen temporal
        os.remove(temp_image_path)

        # Devolver el resultado de la clasificación
        result = "Perro" if confidence > 0.5 else "No perro"
        return {
            "clasificacion": result,
            "confianza": f"{confidence * 100:.2f}%"
        }

    except Exception as e:
        return {"error": str(e)}

# Mensaje de inicio
@app.get("/")
def read_root():
    return {"mensaje": "API de clasificación de mascotas activa."}