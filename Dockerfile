# Imagen base de Python
FROM python:3.11-slim

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para aprovechar caché de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY main_whatsapp.py .
COPY whatsapp_client.py .

# Copiar script de ingesta y documentos (NUEVO)
COPY ingest.py .
COPY documents/ ./documents/

# Ejecutar ingesta para crear knowledge base (NUEVO)
RUN python ingest.py

# Los archivos knowledge_base.index y knowledge_base.json
# se crearán durante el build

# Crear usuario no-root para seguridad
RUN useradd -m -u 1001 chatbot && \
    chown -R chatbot:chatbot /app

# Cambiar a usuario no-root
USER chatbot

# Health check (verifica que puede conectarse a la API)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('${WHATSAPP_API_URL}/health', timeout=5)" || exit 1

# Comando de inicio
CMD ["python", "-u", "main.py"]