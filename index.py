from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite  # Usar tflite-runtime para cargar el modelo

app = FastAPI()

# Configurar CORS para permitir solicitudes desde cualquier origen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir la ruta al modelo TFLite
MODEL_PATH = "TallerPerritos.tflite"

# Cargar el modelo TFLite
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Obtener detalles de entrada y salida del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Modelo TFLite cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo TFLite: {e}")
    interpreter = None


def load_and_preprocess_image(image_path):
    """
    Carga y preprocesa la imagen para que sea compatible con TensorFlow Lite.
    """
    img = Image.open(image_path).convert('RGB')  # Convertir a RGB
    img = img.resize((224, 224))  # Redimensionar a 224x224
    img_array = np.array(img, dtype=np.float32)  # Convertir a arreglo de tipo float32
    img_array = img_array / 127.5 - 1.0  # Escalar píxeles al rango [-1, 1]
    return np.expand_dims(img_array, axis=0)  # Agregar una dimensión para lotes


@app.post("/classify_pet")
async def classify_pet(file: UploadFile = File(...)):
    try:
        # Verificar si el intérprete TFLite está cargado
        if interpreter is None:
            return {"error": "El modelo TFLite no se cargó correctamente."}

        # Guardar la imagen temporalmente
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocesar la imagen
        processed_image = load_and_preprocess_image(temp_image_path)

        # Establecer la entrada del modelo
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Ejecutar la inferencia
        interpreter.invoke()

        # Obtener el resultado de salida
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = output_data[0][0]

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


@app.get("/")
def read_root():
    return {"mensaje": "API de clasificación de mascotas con TFLite activa."}
