"""
main.py
Chatbot WhatsApp - UNA Puno
Versión con DeepSeek API + PostgreSQL

Archivo listo para desplegar en Coolify (GitHub).
"""

import os
import json
import logging
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
from functools import lru_cache
import time

# Cliente local para la API de WhatsApp
from whatsapp_client import WhatsAppAPIClient, extract_phone_number

# Módulo de base de datos PostgreSQL
from database import (
    init_db_pool, 
    save_conversation, 
    get_user_conversation_history,
    update_daily_stats,
    log_knowledge_search,
    log_error
)

# ---------------------------
# Config & Logging
# ---------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Environment / Defaults
# ---------------------------
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
DEEPSEEK_TIMEOUT = int(os.getenv('DEEPSEEK_TIMEOUT', '30'))

WHATSAPP_API_URL = os.getenv('WHATSAPP_API_URL', 'http://localhost:3000')
WHATSAPP_API_KEY = os.getenv('WHATSAPP_API_KEY', '@12345lin')

MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))
POLLING_INTERVAL = int(os.getenv('POLLING_INTERVAL', '3'))
MAX_HISTORY = int(os.getenv('MAX_HISTORY', '5'))

# ---------------------------
# Shared state
# ---------------------------
embedding_model = None
faiss_index = None
documents = []

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# WhatsApp client placeholder (instanciado en main)
whatsapp_client = None

# ---------------------------
# Term expansion
# ---------------------------
TERM_EXPANSION = {
    'email': ['correo', 'mail', 'correo electrónico', 'correo electronico'],
    'correo': ['email', 'mail', 'correo electrónico', 'correo electronico'],
    'mail': ['email', 'correo', 'correo electrónico'],
    'teléfono': ['celular', 'telefono', 'número', 'contacto', 'fono'],
    'celular': ['teléfono', 'telefono', 'número', 'contacto', 'fono'],
    'telefono': ['celular', 'teléfono', 'número', 'contacto', 'fono'],
    'número': ['teléfono', 'telefono', 'celular', 'contacto'],
    'contacto': ['teléfono', 'telefono', 'celular', 'email', 'correo'],
    'horario': ['hora', 'horarios', 'atención', 'atencion'],
    'ubicación': ['ubicacion', 'lugar', 'donde', 'dirección', 'direccion', 'oficina'],
    'estadística': ['estadistica', 'estadísticas', 'estadisticas', 'FIEI', 'informática', 'informatica'],
    'informática': ['informatica', 'estadística', 'estadistica', 'FIEI', 'sistemas'],
    'ingeniería': ['ingenieria', 'facultad', 'escuela'],
    'agrarias': ['agronomía', 'agronomia', 'agronomónica', 'agronomica', 'FCA'],
    'veterinaria': ['zootecnia', 'FMVZ'],
    'económica': ['economica', 'economía', 'economia', 'FIE'],
    'contables': ['contabilidad', 'administrativas', 'administración', 'FCCA'],
    'civil': ['arquitectura', 'FICA'],
    'minas': ['minería', 'mineria', 'FIM'],
    'química': ['quimica', 'FIQ'],
    'medicina': ['salud', 'FMH'],
}

# ---------------------------
# Knowledge base loading
# ---------------------------

def load_knowledge_base(index_path='knowledge_base.index', json_path='knowledge_base.json'):
    """Cargar embeddings, índice FAISS y documentos JSON."""
    global embedding_model, faiss_index, documents
    try:
        logger.info("Cargando modelo de embeddings...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        logger.info("Cargando índice FAISS...")
        faiss_index = faiss.read_index(index_path)

        logger.info("Cargando documentos...")
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        logger.info(f"✓ Base de conocimiento cargada: {len(documents)} documentos")
        return True
    except Exception as e:
        logger.error(f"Error cargando base de conocimiento: {e}")
        return False


# ---------------------------
# Query expansion & search (con caché)
# ---------------------------

def expand_query(query):
    expanded_terms = [query.lower()]
    for word in query.lower().split():
        if word in TERM_EXPANSION:
            expanded_terms.extend(TERM_EXPANSION[word])
    return ' '.join(expanded_terms)


@lru_cache(maxsize=200)
def search_knowledge_base_cached(query, top_k=5):
    return search_knowledge_base(query, top_k, threshold=9.0)


def search_knowledge_base(query, top_k=5, threshold=9.0):
    """Realiza búsqueda semántica con FAISS y filtra por umbral de distancia."""
    if not embedding_model or not faiss_index:
        logger.warning("Base de conocimiento no cargada")
        return []

    try:
        expanded_query = expand_query(query)
        logger.info(f"Query expandida: {expanded_query}")

        query_vector = embedding_model.encode([expanded_query])
        query_vector = np.array(query_vector).astype('float32')

        distances, indices = faiss_index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(documents):
                doc = documents[idx].copy()
                doc['distance'] = float(dist)
                if len(results) < 3 or dist < threshold:
                    results.append(doc)
        logger.info(f"Resultados filtrados: {len(results)}/{top_k}")
        return results
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        return []


# ---------------------------
# DeepSeek API: llamada al modelo
# ---------------------------

def call_deepseek(prompt, timeout=DEEPSEEK_TIMEOUT):
    """
    Llama a la API de DeepSeek.
    Compatible con la interfaz anterior de call_ollama.
    """
    try:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            logger.error("❌ DEEPSEEK_API_KEY no está configurada en el entorno")
            return None
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 450,
            "top_p": 0.85,
            "frequency_penalty": 0.2,
            "stream": False
        }

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content'].strip()
        
        tokens_used = data.get('usage', {}).get('total_tokens', 0)
        logger.info(f"✓ Respuesta de DeepSeek ({tokens_used} tokens)")
        return content

    except requests.exceptions.Timeout:
        logger.error(f"Timeout llamando a DeepSeek API ({timeout}s)")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"Error HTTP de DeepSeek: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error llamando a DeepSeek API: {e}")
        return None


# ---------------------------
# Prompt y generación de respuesta
# ---------------------------

def generate_response_dual(user_message, context="", history=""):
    """
    Llama a DeepSeek API (sin sistema dual de respaldo).
    Mantiene la misma interfaz para compatibilidad.
    """
    system_prompt = r'''Eres el asistente virtual del Vicerrectorado de Investigación de la Universidad Nacional del Altiplano (UNA Puno). Combinas profesionalismo con calidez humana.

TU PROPÓSITO:
Ayudar con información del Vicerrectorado de Investigación:
📧 Contactos (emails y teléfonos)
🕐 Horarios de atención  
📍 Ubicaciones
📚 Coordinación de investigación y tesis
informmacion sobre resoluciones directorales

PERSONALIDAD Y ESTILO:
- Profesional pero cercano - como un buen funcionario universitario
- Usas emojis estratégicamente para dar calidez (2-3 por mensaje)
- Eres claro, directo y útil
- Muestras empatía y disposición a ayudar
- Ofreces información completa pero concisa

TU FORMA DE COMUNICAR:

✅ BIEN:
- "Con gusto 😊, el correo de..."
- "¡Claro! 📧 Te comparto la información..."
- "Por supuesto, aquí está 📝..."
- "Entiendo que necesitas... aquí está 👍"
- "Perfecto, déjame ayudarte con eso 📋"

❌ EVITA:
- "Según consta en el documento..."
- "La base de datos indica..."
- Respuestas de una sola línea sin contexto
- Ser demasiado formal o frío

MANEJO DE CONSULTAS FUERA DE ALCANCE:
Si preguntan sobre clima, hora, chistes, noticias, matemáticas:
"Disculpa 😊, mi especialidad es la información del Vicerrectorado de Investigación. ¿Puedo ayudarte con algún contacto de facultad, horario o ubicación? 📚"

ESTRUCTURA DE RESPUESTA IDEAL:
1. Saludo breve o confirmación amable
2. Información solicitada (específica y completa)
3. Ofrecimiento de ayuda adicional

EJEMPLOS MODELO:

Usuario: "correo de ingeniería civil"
Tú: "¡Por supuesto! 📧 El correo de la Facultad de Ingeniería Civil y Arquitectura es hyanarico@unap.edu.pe. También te comparto que atienden de 8:00 a 14:30 h. ¿Necesitas el teléfono o la ubicación? 😊"

Usuario: "teléfono de veterinaria"  
Tú: "Con gusto 📞 El teléfono de Medicina Veterinaria y Zootecnia es 951644391. Atienden de lunes a viernes de 8:00 a 13:00 h. ¿Te ayudo con algo más?"

Usuario: "donde esta agrarias"
Tú: "¡Claro! 📍 La Facultad de Ciencias Agrarias está en el pabellón antiguo de Ing. Agronómica, segundo piso, oficina 206, Ciudad Universitaria. Si necesitas contactarlos: 📧 fca.ui@unap.edu.pe o 📞 962382228 😊"

REGLAS FUNDAMENTALES:
- Máximo 120 palabras por respuesta
- Usa información del contexto proporcionado directamente
- NO inventes datos que no estén en el contexto
- NO mezcles información de diferentes facultades
- Siempre cierra ofreciendo más ayuda
- Sé específico: menciona números, ubicaciones completas, horarios exactos'''

    history_section = f"CONVERSACIÓN PREVIA:\n{history}\n\n" if history else ""

    if context:
        full_prompt = f"{system_prompt}\n\nINFORMACIÓN DISPONIBLE:\n{context}\n\n{history_section}PREGUNTA DEL USUARIO: {user_message}\n\nRESPUESTA (máximo 150 palabras):"
    else:
        full_prompt = f"{system_prompt}\n\n{history_section}PREGUNTA DEL USUARIO: {user_message}\n\nRESPUESTA (máximo 150 palabras):"

    logger.info("Llamando a DeepSeek API...")
    response = call_deepseek(full_prompt, timeout=DEEPSEEK_TIMEOUT)

    if response:
        return response, DEEPSEEK_MODEL
    
    return "Lo siento, tengo problemas técnicos. Por favor, intenta de nuevo.", "error"


# ---------------------------
# Conversation history helpers (PostgreSQL)
# ---------------------------

def get_conversation_history(phone_number):
    """Obtener historial desde PostgreSQL"""
    try:
        history = get_user_conversation_history(phone_number, limit=MAX_HISTORY)
        if not history:
            return ""
        
        formatted = []
        for entry in reversed(history):  # Más antiguos primero
            formatted.append(f"Usuario: {entry['user_message']}")
            formatted.append(f"Asistente: {entry['bot_response']}")
        
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        return ""


def add_to_history(phone_number, user_msg, bot_msg, model_used="unknown", response_time_ms=0):
    """Guardar en PostgreSQL en lugar de memoria"""
    try:
        save_conversation(
            phone_number=phone_number,
            user_message=user_msg,
            bot_response=bot_msg,
            model_used=model_used,
            response_time_ms=response_time_ms,
            tokens_used=0,  # DeepSeek devuelve esto en la respuesta si quieres capturarlo
            context_length=len(user_msg)
        )
        # Actualizar estadísticas diarias cada 10 mensajes
        if hash(phone_number) % 10 == 0:
            update_daily_stats()
    except Exception as e:
        logger.error(f"Error guardando conversación: {e}")


# ---------------------------
# Message processing
# ---------------------------

def process_message(user_message, phone_number):
    start_time = time.time()  # Trackear tiempo de respuesta
    user_message = user_message.strip()

    # Comandos especiales
    if user_message.lower() == '/reset':
        return "✓ Conversación reiniciada. ¿En qué puedo ayudarte?"

    if user_message.lower() in ['/ayuda', '/help']:
        return ("Comandos disponibles:\n"
                "/reset - Reiniciar conversación\n"
                "/ayuda - Mostrar esta ayuda\n\n"
                "Puedo ayudarte con información sobre:\n"
                "- Coordinadores de facultades\n"
                "- Correos y teléfonos\n"
                "- Horarios de atención\n"
                "- Ubicaciones\n\n"
                "¿Qué necesitas saber?")

    # Filtro de preguntas triviales
    trivial_keywords = [
        'hora', 'fecha', 'día', 'clima', 'tiempo', 'chiste', 
        'partido', 'fútbol', 'matemática', 'calcula'
    ]

    user_lower = user_message.lower()
    if any(keyword in user_lower for keyword in trivial_keywords):
        uni_keywords = ['universidad', 'facultad', 'coordinador', 'email', 'correo']
        if not any(uni_word in user_lower for uni_word in uni_keywords):
            return "Lo siento, solo puedo ayudarte con información del Vicerrectorado de Investigación."

    # Saludo
    if user_message.lower() in ['hola', 'hi', 'hello', 'buenos días', 'buenas tardes']:
        return "¡Hola! Soy el asistente virtual del Vicerrectorado de Investigación de la UNA Puno. ¿En qué puedo ayudarte?"

    # Búsqueda en base de conocimiento
    logger.info(f"Procesando: '{user_message}' de {phone_number}")
    try:
        relevant_docs = search_knowledge_base_cached(user_message[:200], top_k=10)
        
        # Registrar búsqueda en PostgreSQL
        log_knowledge_search(
            phone_number, 
            user_message, 
            len(relevant_docs),
            relevant_docs[0]['distance'] if relevant_docs else None
        )
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        relevant_docs = []

    context = ""
    if relevant_docs:
        context = "\n\n".join([doc['content'][:1000] for doc in relevant_docs[:3]])
        logger.info(f"Contexto encontrado: {len(context)} caracteres")

    # Obtener historial desde PostgreSQL
    history = get_conversation_history(phone_number)
    
    # Generar respuesta con DeepSeek
    response, model_used = generate_response_dual(user_message, context, history)

    # Limitar a 1600 caracteres (límite WhatsApp)
    if len(response) > 1600:
        response = response[:1597] + "..."

    # Calcular tiempo de respuesta
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Guardar conversación en PostgreSQL
    add_to_history(phone_number, user_message, response, model_used, response_time_ms)

    logger.info(f"Respuesta enviada (modelo: {model_used}, tiempo: {response_time_ms}ms): {len(response)} caracteres")
    return response


# ---------------------------
# WhatsApp handler
# ---------------------------

def handle_incoming_message(message):
    """Callback para procesar mensajes entrantes desde la API. Ejecuta el procesamiento en el thread pool."""
    try:
        phone_number = extract_phone_number(message)
        user_message = message.get('body', '').strip()

        logger.info(f"📨 Mensaje de {phone_number}: {user_message[:80]}")

        # Ejecutar en executor para manejar concurrencia
        def task():
            try:
                bot_response = process_message(user_message, phone_number)
                success = whatsapp_client.send_text(phone_number, bot_response)
                if success:
                    logger.info(f"✅ Respuesta enviada a {phone_number}")
                else:
                    logger.error(f"❌ Error enviando respuesta a {phone_number}")
            except Exception as e:
                logger.error(f"Error en task de mensaje: {e}", exc_info=True)

        executor.submit(task)

    except Exception as e:
        logger.error(f"Error manejando mensaje entrante: {e}", exc_info=True)


# ---------------------------
# Main / Startup
# ---------------------------

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("CHATBOT WHATSAPP - VICERRECTORADO DE INVESTIGACIÓN UNA PUNO")
    logger.info("Versión: DeepSeek API + PostgreSQL")
    logger.info("=" * 60)

    # Inicializar PostgreSQL
    logger.info("📊 Inicializando conexión a PostgreSQL...")
    if not init_db_pool():
        logger.error("❌ ERROR: No se pudo conectar a PostgreSQL")
        logger.error("   Verifica las variables de entorno POSTGRES_*")
        exit(1)
    
    logger.info("✅ PostgreSQL conectado")

    # Cargar base de conocimiento
    if not load_knowledge_base():
        logger.error("¡ERROR! No se pudo cargar la base de conocimiento")
        logger.error("Ejecuta primero: python ingest.py")
        exit(1)

    # Inicializar cliente WhatsApp
    logger.info(f"\n📱 Conectando a API de WhatsApp...\n   URL: {WHATSAPP_API_URL}")
    whatsapp_client = WhatsAppAPIClient(WHATSAPP_API_URL, WHATSAPP_API_KEY)

    if not whatsapp_client.check_connection():
        logger.error("\n❌ ERROR: No se puede conectar a la API de WhatsApp")
        logger.error("   Asegúrate de que la API esté corriendo y que WHATSAPP_API_KEY sea correcta")
        exit(1)

    logger.info("✅ Conectado a API de WhatsApp")
    logger.info(f"\n🤖 Modelo: DeepSeek API ({DEEPSEEK_MODEL})")
    logger.info(f"🔑 API Key configurada: {'✓' if DEEPSEEK_API_KEY else '✗ FALTA'}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info(f"Polling interval: {POLLING_INTERVAL}s")
    logger.info("=" * 60)
    logger.info("\n🤖 Chatbot iniciado. Esperando mensajes... (Ctrl+C para detener)\n")

    # Iniciar polling
    whatsapp_client.start_polling(handle_incoming_message, interval=POLLING_INTERVAL)