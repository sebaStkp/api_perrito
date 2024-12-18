# Use la imagen base de Python 3.9
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto
COPY . .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar la aplicación
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]