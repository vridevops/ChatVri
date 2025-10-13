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

# Cliente local para la API de WhatsApp (interfaz mínima esperada)
from whatsapp_client import WhatsAppAPIClient, extract_phone_number

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
# ✅ NUEVAS VARIABLES PARA DEEPSEEK
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
OLLAMA_TIMEOUT = int(os.getenv('DEEPSEEK_TIMEOUT', '30'))  # Renombrar internamente o mantener compatible

WHATSAPP_API_URL = os.getenv('WHATSAPP_API_URL', 'http://localhost:3000')
WHATSAPP_API_KEY = os.getenv('WHATSAPP_API_KEY', '@12345lin')

MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))
POLLING_INTERVAL = int(os.getenv('POLLING_INTERVAL', '3'))
MAX_HISTORY = int(os.getenv('MAX_HISTORY', '5'))

# ---------------------------
# Shared state
# ---------------------------
user_conversations = defaultdict(list)        # phone -> list of exchanges
user_locks = defaultdict(threading.Lock)      # phone -> lock

embedding_model = None
faiss_index = None
documents = []

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# WhatsApp client placeholder (instanciado en main)
whatsapp_client = None

# ---------------------------
# Term expansion (usar versión extendida del primer código)
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


def call_deepseek(prompt, timeout=OLLAMA_TIMEOUT):
    """
    Llama a la API de DeepSeek.
    Compatible con la interfaz anterior de call_ollama.
    """
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
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
        
        logger.info(f"✓ Respuesta de DeepSeek ({data.get('usage', {}).get('total_tokens', 0)} tokens)")
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
# Promp
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
    response = call_deepseek(full_prompt, timeout=OLLAMA_TIMEOUT)

    if response:
        return response, "deepseek-chat"
    
    return "Lo siento, tengo problemas técnicos. Por favor, intenta de nuevo.", "error"

# ---------------------------
# Conversation history helpers (thread-safe)
# ---------------------------

def get_conversation_history(phone_number):
    with user_locks[phone_number]:
        history = user_conversations[phone_number]
        if not history:
            return ""

        formatted = []
        for entry in history[-MAX_HISTORY:]:
            formatted.append(f"Usuario: {entry['user']}")
            formatted.append(f"Asistente: {entry['bot']}")

        return "\n".join(formatted)


def add_to_history(phone_number, user_msg, bot_msg):
    with user_locks[phone_number]:
        user_conversations[phone_number].append({
            'user': user_msg,
            'bot': bot_msg,
            'timestamp': datetime.now().isoformat()
        })

        if len(user_conversations[phone_number]) > MAX_HISTORY:
            user_conversations[phone_number] = user_conversations[phone_number][-MAX_HISTORY:]

    save_stats_to_file()


# ---------------------------
# Stats & persistence
# ---------------------------

def save_stats_to_file(path='dashboard_stats.json'):
    try:
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_users': len(user_conversations),
            'total_messages': sum(len(convs) for convs in user_conversations.values()),
            'kb_documents': len(documents),
            'model_used': DEEPSEEK_MODEL if DEEPSEEK_MODEL else 'deepseek-chat',
            'conversations': []
        }

        for phone, convs in user_conversations.items():
            for conv in convs:
                stats['conversations'].append({
                    'phone': phone,
                    'timestamp': conv['timestamp'],
                    'user': conv['user'],
                    'bot': conv['bot']
                })

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error guardando estadísticas: {e}")


# ---------------------------
# Message processing (combina lo mejor de ambos códigos)
# ---------------------------

def process_message(user_message, phone_number):
    user_message = user_message.strip()

    # Comandos especiales
    if user_message.lower() == '/reset':
        with user_locks[phone_number]:
            user_conversations[phone_number] = []
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

    # Búsqueda en base de conocimiento (usar cache cuando aplique)
    logger.info(f"Procesando: '{user_message}' de {phone_number}")
    try:
        relevant_docs = search_knowledge_base_cached(user_message[:200], top_k=10)
    except Exception:
        relevant_docs = []

    context = ""
    if relevant_docs:
        context = "\n\n".join([doc['content'][:1000] for doc in relevant_docs[:3]])
        logger.info(f"Contexto encontrado: {len(context)} caracteres")

    history = get_conversation_history(phone_number)
    response, model_used = generate_response_dual(user_message, context, history)

    # Limitar a 1600 caracteres (límite WhatsApp)
    if len(response) > 1600:
        response = response[:1597] + "..."

    add_to_history(phone_number, user_message, response)

    logger.info(f"Respuesta enviada (modelo: {model_used}): {len(response)} caracteres")
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
    logger.info("Versión: Unificada (estable + optimizada)")
    logger.info("=" * 60)

    # Cargar base de conocimiento
    if not load_knowledge_base():
        logger.error("¡ERROR! No se pudo cargar la base de conocimiento")
        logger.error("Ejecuta primero: python ingest.py")
        exit(1)

    # Inicializar estadísticas
    logger.info("Inicializando archivo de estadísticas...")
    save_stats_to_file()

    # Inicializar cliente WhatsApp
    logger.info(f"\n📱 Conectando a API de WhatsApp...\n   URL: {WHATSAPP_API_URL}")
    whatsapp_client = WhatsAppAPIClient(WHATSAPP_API_URL, WHATSAPP_API_KEY)

    if not whatsapp_client.check_connection():
        logger.error("\n❌ ERROR: No se puede conectar a la API de WhatsApp")
        logger.error("   Asegúrate de que la API esté corriendo y que WHATSAPP_API_KEY sea correcta")
        exit(1)

    logger.info("✅ Conectado a API de WhatsApp")
    # ✅ NUEVO:
    logger.info(f"\n🤖 Modelo: DeepSeek API ({DEEPSEEK_MODEL})")
    logger.info(f"🔑 API Key configurada: {'✓' if DEEPSEEK_API_KEY else '✗ FALTA'}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info(f"Polling interval: {POLLING_INTERVAL}s")
    logger.info("=" * 60)
    logger.info("\n🤖 Chatbot iniciado. Esperando mensajes... (Ctrl+C para detener)\n")

    # Iniciar polling
    whatsapp_client.start_polling(handle_incoming_message, interval=POLLING_INTERVAL)
