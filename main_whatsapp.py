"""
Chatbot WhatsApp - Vicerrectorado de Investigación UNA Puno
Versión adaptada para API de WhatsApp (sin Twilio)
Sistema dual de IA con Ollama (deepseek-v3.1:671b-cloud + gemma3:1b)
"""

import os
import json
import logging
from datetime import datetime
from collections import defaultdict
import requests
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ⭐ IMPORTAR CLIENTE DE WHATSAPP
from whatsapp_client import WhatsAppAPIClient, extract_phone_number

# Configuración
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables de entorno
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-v3.1:671b-cloud')
OLLAMA_MODEL_BACKUP = os.getenv('OLLAMA_MODEL_BACKUP', 'gemma3:1b')

# ⭐ NUEVAS VARIABLES PARA LA API DE WHATSAPP
WHATSAPP_API_URL = os.getenv('WHATSAPP_API_URL', 'http://localhost:3000')
WHATSAPP_API_KEY = os.getenv('WHATSAPP_API_KEY', '@12345lin')

# Memoria conversacional
user_conversations = defaultdict(list)
MAX_HISTORY = 5

# Modelo de embeddings
embedding_model = None
faiss_index = None
documents = []

# ⭐ CLIENTE DE WHATSAPP (REEMPLAZA TWILIO)
whatsapp_client = None

# Expansión de términos para búsqueda mejorada
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
    'agrarias': ['agronomía', 'agronomia', 'agronómica', 'agronomica', 'FCA'],
    'veterinaria': ['zootecnia', 'FMVZ'],
    'económica': ['economica', 'economía', 'economia', 'FIE'],
    'contables': ['contabilidad', 'administrativas', 'administración', 'FCCA'],
    'civil': ['arquitectura', 'FICA'],
    'minas': ['minería', 'mineria', 'FIM'],
    'química': ['quimica', 'FIQ'],
    'medicina': ['salud', 'FMH'],
}


def load_knowledge_base():
    """Cargar base de conocimiento FAISS"""
    global embedding_model, faiss_index, documents
    
    try:
        logger.info("Cargando modelo de embeddings...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        logger.info("Cargando índice FAISS...")
        faiss_index = faiss.read_index('knowledge_base.index')
        
        logger.info("Cargando documentos...")
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"✓ Base de conocimiento cargada: {len(documents)} documentos")
        return True
    except Exception as e:
        logger.error(f"Error cargando base de conocimiento: {e}")
        return False


def expand_query(query):
    """Expandir términos de búsqueda"""
    expanded_terms = [query.lower()]
    
    for word in query.lower().split():
        if word in TERM_EXPANSION:
            expanded_terms.extend(TERM_EXPANSION[word])
    
    return ' '.join(expanded_terms)


def search_knowledge_base(query, top_k=5, threshold=9.0):
    """Búsqueda semántica mejorada en FAISS"""
    if not embedding_model or not faiss_index:
        logger.warning("Base de conocimiento no cargada")
        return []
    
    try:
        expanded_query = expand_query(query)
        logger.info(f"Query expandida: {expanded_query}")
        
        query_vector = embedding_model.encode([expanded_query])
        query_vector = np.array(query_vector).astype('float32')
        
        distances, indices = faiss_index.search(query_vector, top_k)
        
        logger.info("=== RESULTADOS DE BÚSQUEDA ===")
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(documents):
                logger.info(f"  Distancia: {dist:.3f} | {documents[idx]['title'][:60]}")
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(documents):
                doc = documents[idx].copy()
                doc['distance'] = float(dist)
                
                if len(results) < 3 or dist < threshold:
                    results.append(doc)
                    logger.info(f"  → ACEPTADO: {doc['title'][:50]}... (distancia: {dist:.3f})")
        
        logger.info(f"Resultados filtrados: {len(results)}/{top_k}")
        return results
        
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        return []


def call_ollama(prompt, model_name, stream=True):
    """Llamar a Ollama con streaming"""
    try:
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_predict": 450,
                "top_p": 0.85,
                "top_k": 5,
                "num_ctx": 3072,
                "repeat_penalty": 1.2
            }
        }
        
        response = requests.post(url, json=payload, stream=stream, timeout=60)
        response.raise_for_status()
        
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response += data['response']
                    if data.get('done', False):
                        break
            return full_response.strip()
        else:
            return response.json().get('response', '').strip()
            
    except Exception as e:
        logger.error(f"Error llamando a {model_name}: {e}")
        return None


def generate_response_dual(user_message, context="", history=""):
    """Sistema dual de IA: deepseek principal, gemma3 respaldo"""
    
    system_prompt = """Eres el asistente virtual del Vicerrectorado de Investigación de la Universidad Nacional del Altiplano (UNA Puno). Combinas profesionalismo con calidez humana.

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
- Sé específico: menciona números, ubicaciones completas, horarios exactos"""

    if context:
        full_prompt = f"""{system_prompt}

INFORMACIÓN DISPONIBLE:
{context}

{"CONVERSACIÓN PREVIA:\n" + history if history else ""}

PREGUNTA DEL USUARIO: {user_message}

RESPUESTA (máximo 150 palabras):"""
    else:
        full_prompt = f"""{system_prompt}

{"CONVERSACIÓN PREVIA:\n" + history if history else ""}

PREGUNTA DEL USUARIO: {user_message}

RESPUESTA (máximo 150 palabras):"""

    logger.info(f"Intentando con {OLLAMA_MODEL}...")
    response = call_ollama(full_prompt, OLLAMA_MODEL)
    
    if response:
        logger.info(f"✓ Respuesta de {OLLAMA_MODEL}")
        return response, OLLAMA_MODEL
    
    logger.warning(f"{OLLAMA_MODEL} falló, usando {OLLAMA_MODEL_BACKUP}...")
    response = call_ollama(full_prompt, OLLAMA_MODEL_BACKUP)
    
    if response:
        logger.info(f"✓ Respuesta de {OLLAMA_MODEL_BACKUP}")
        return response, OLLAMA_MODEL_BACKUP
    
    return "Lo siento, tengo problemas técnicos. Por favor, intenta de nuevo.", "error"


def get_conversation_history(phone_number):
    """Obtener historial de conversación"""
    history = user_conversations[phone_number]
    if not history:
        return ""
    
    formatted = []
    for entry in history[-MAX_HISTORY:]:
        formatted.append(f"Usuario: {entry['user']}")
        formatted.append(f"Asistente: {entry['bot']}")
    
    return "\n".join(formatted)


def add_to_history(phone_number, user_msg, bot_msg):
    """Agregar interacción al historial"""
    user_conversations[phone_number].append({
        'user': user_msg,
        'bot': bot_msg,
        'timestamp': datetime.now().isoformat()
    })
    
    if len(user_conversations[phone_number]) > MAX_HISTORY:
        user_conversations[phone_number] = user_conversations[phone_number][-MAX_HISTORY:]
    
    save_stats_to_file()


def process_message(user_message, phone_number):
    """Procesar mensaje del usuario"""
    user_message = user_message.strip()
    
    # Comandos especiales
    if user_message.lower() == '/reset':
        user_conversations[phone_number] = []
        return "✓ Conversación reiniciada. ¿En qué puedo ayudarte?"
    
    if user_message.lower() in ['/ayuda', '/help']:
        return """Comandos disponibles:
/reset - Reiniciar conversación
/ayuda - Mostrar esta ayuda

Puedo ayudarte con información sobre:
- Coordinadores de facultades
- Correos y teléfonos
- Horarios de atención
- Ubicaciones

¿Qué necesitas saber?"""
    
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
    relevant_docs = search_knowledge_base(user_message, top_k=10, threshold=9.0)
    
    context = ""
    if relevant_docs:
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        logger.info(f"Contexto encontrado: {len(context)} caracteres")
    
    history = get_conversation_history(phone_number)
    response, model_used = generate_response_dual(user_message, context, history)
    
    # Limitar a 1600 caracteres (límite WhatsApp)
    if len(response) > 1600:
        response = response[:1597] + "..."
    
    add_to_history(phone_number, user_message, response)
    
    logger.info(f"Respuesta enviada (modelo: {model_used}): {len(response)} caracteres")
    return response


def save_stats_to_file():
    """Guardar estadísticas en archivo JSON"""
    try:
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_users': len(user_conversations),
            'total_messages': sum(len(convs) for convs in user_conversations.values()),
            'kb_documents': len(documents),
            'model_used': OLLAMA_MODEL.split(':')[0] if ':' in OLLAMA_MODEL else OLLAMA_MODEL,
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
        
        with open('dashboard_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error guardando estadísticas: {e}")


# ⭐ NUEVA FUNCIÓN: Manejador de mensajes de WhatsApp
def handle_incoming_message(message):
    """
    Callback para procesar mensajes entrantes desde la API
    
    Args:
        message: Dict con datos del mensaje de la API
    """
    try:
        # Extraer datos del mensaje
        phone_number = extract_phone_number(message)
        user_message = message.get('body', '').strip()
        
        logger.info(f"📨 Mensaje de {phone_number}: {user_message}")
        
        # Procesar con la lógica del chatbot
        bot_response = process_message(user_message, phone_number)
        
        # Enviar respuesta usando la API de WhatsApp
        success = whatsapp_client.send_text(phone_number, bot_response)
        
        if success:
            logger.info(f"✅ Respuesta enviada a {phone_number}")
        else:
            logger.error(f"❌ Error enviando respuesta a {phone_number}")
            
    except Exception as e:
        logger.error(f"Error manejando mensaje: {e}", exc_info=True)

async def send_media_url(self, phone: str, media_url: str, caption: str = "") -> bool:
    """Enviar archivo por URL"""
    try:
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.api_url}/api/whatsapp/send/media-url"
        
        payload = {
            'phone': phone,
            'mediaUrl': media_url,
            'caption': caption
        }
        
        async with self.session.post(url, headers=headers, json=payload, timeout=30) as resp:
            success = resp.status == 200
            if success:
                logger.info(f"✅ Archivo enviado a {phone}")
            else:
                logger.error(f"❌ Error enviando archivo: {resp.status}")
            return success
    
    except Exception as e:
        logger.error(f"Error en send_media_url: {e}")
        return False

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("CHATBOT WHATSAPP - VICERRECTORADO DE INVESTIGACIÓN UNA PUNO")
    logger.info("Versión con API de WhatsApp (sin Twilio)")
    logger.info("=" * 60)
    
    # Cargar base de conocimiento
    if not load_knowledge_base():
        logger.error("¡ERROR! No se pudo cargar la base de conocimiento")
        logger.error("Ejecuta primero: python ingest.py")
        exit(1)
    
    # Inicializar estadísticas
    logger.info("Inicializando archivo de estadísticas...")
    save_stats_to_file()
    
    # ⭐ CREAR CLIENTE DE WHATSAPP (REEMPLAZA TWILIO)
    logger.info(f"\n📱 Conectando a API de WhatsApp...")
    logger.info(f"   URL: {WHATSAPP_API_URL}")
    
    whatsapp_client = WhatsAppAPIClient(WHATSAPP_API_URL, WHATSAPP_API_KEY)
    
    # Verificar conexión
    if not whatsapp_client.check_connection():
        logger.error("\n❌ ERROR: No se puede conectar a la API de WhatsApp")
        logger.error("   Asegúrate de que:")
        logger.error("   1. La API esté corriendo: npm run dev")
        logger.error("   2. WhatsApp esté autenticado y conectado")
        logger.error(f"   3. WHATSAPP_API_URL={WHATSAPP_API_URL}")
        logger.error(f"   4. WHATSAPP_API_KEY={WHATSAPP_API_KEY}")
        exit(1)
    
    logger.info("✅ Conectado a API de WhatsApp")
    logger.info(f"\nModelo principal: {OLLAMA_MODEL}")
    logger.info(f"Modelo respaldo: {OLLAMA_MODEL_BACKUP}")
    logger.info("=" * 60)
    logger.info("\n🤖 Chatbot iniciado. Esperando mensajes...")
    logger.info("   Presiona Ctrl+C para detener\n")
    
    # ⭐ INICIAR POLLING (REEMPLAZA WEBHOOK DE FLASK)
    whatsapp_client.start_polling(handle_incoming_message, interval=2)