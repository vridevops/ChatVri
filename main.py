"""
main_async.py
Chatbot WhatsApp - UNA Puno
Versión ASYNC optimizada - Compatible con WhatsAppAPIClient existente
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
import aiohttp
import asyncpg
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import time
import threading

from whatsapp_client import WhatsAppAPIClient, extract_phone_number

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Environment
# ---------------------------
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
DEEPSEEK_TIMEOUT = int(os.getenv('DEEPSEEK_TIMEOUT', '20'))

WHATSAPP_API_URL = os.getenv('WHATSAPP_API_URL')
WHATSAPP_API_KEY = os.getenv('WHATSAPP_API_KEY')

MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT', '100'))
POLLING_INTERVAL = int(os.getenv('POLLING_INTERVAL', '2'))
MAX_HISTORY = int(os.getenv('MAX_HISTORY', '5'))
INACTIVITY_TIMEOUT = int(os.getenv('INACTIVITY_TIMEOUT', '1800'))

# State
db_pool = None
semaphore = asyncio.Semaphore(MAX_CONCURRENT)
http_session = None
embedding_model = None
faiss_index = None
documents = []
whatsapp_client = None
event_loop = None

# Tracking
user_last_activity = {}
user_closed_sessions = set()

# ---------------------------
# Term expansion
# ---------------------------
TERM_EXPANSION = {
    'email': ['correo', 'mail', 'correo electrónico'],
    'correo': ['email', 'mail'],
    'teléfono': ['celular', 'telefono', 'número', 'contacto'],
    'celular': ['teléfono', 'telefono', 'número'],
    'horario': ['hora', 'horarios', 'atención'],
    'ubicación': ['ubicacion', 'lugar', 'donde', 'dirección'],
}

# ---------------------------
# Database async
# ---------------------------

async def init_db_pool_async():
    """Inicializar pool async de PostgreSQL"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DB', 'postgres'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            min_size=10,
            max_size=100,
            command_timeout=10
        )
        logger.info("✅ PostgreSQL async pool inicializado")
        return True
    except Exception as e:
        logger.error(f"❌ Error PostgreSQL: {e}")
        return False

async def save_conversation_async(phone, user_msg, bot_msg, model, response_time):
    """Guardar conversación async - NO BLOQUEANTE"""
    try:
        async with db_pool.acquire() as conn:
            user_id = await conn.fetchval("""
                INSERT INTO users (phone_number)
                VALUES ($1)
                ON CONFLICT (phone_number) 
                DO UPDATE SET last_seen = CURRENT_TIMESTAMP
                RETURNING id
            """, phone)
            
            await conn.execute("""
                INSERT INTO conversations 
                (user_id, phone_number, user_message, bot_response, 
                 model_used, response_time_ms, context_length)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, user_id, phone, user_msg, bot_msg, model, response_time, len(user_msg))
            
    except Exception as e:
        logger.error(f"❌ Error guardando: {e}")

async def get_conversation_history_async(phone):
    """Obtener historial async"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT user_message, bot_response
                FROM conversations
                WHERE phone_number = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, phone, MAX_HISTORY)
            
            if not rows:
                return ""
            
            formatted = []
            for row in reversed(rows):
                formatted.append(f"Usuario: {row['user_message']}")
                formatted.append(f"Asistente: {row['bot_response']}")
            
            return "\n".join(formatted)
    except Exception as e:
        logger.error(f"Error historial: {e}")
        return ""

# ---------------------------
# Knowledge base
# ---------------------------

def load_knowledge_base(index_path='knowledge_base.index', json_path='knowledge_base.json'):
    global embedding_model, faiss_index, documents
    try:
        logger.info("Cargando embeddings...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        faiss_index = faiss.read_index(index_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"✓ Base cargada: {len(documents)} docs")
        return True
    except Exception as e:
        logger.error(f"Error KB: {e}")
        return False

def expand_query(query):
    expanded_terms = [query.lower()]
    for word in query.lower().split():
        if word in TERM_EXPANSION:
            expanded_terms.extend(TERM_EXPANSION[word])
    return ' '.join(expanded_terms)

@lru_cache(maxsize=500)
def search_knowledge_base_cached(query, top_k=5):
    return search_knowledge_base(query, top_k)

def search_knowledge_base(query, top_k=5):
    if not embedding_model or not faiss_index:
        return []
    try:
        expanded_query = expand_query(query)
        query_vector = embedding_model.encode([expanded_query])
        query_vector = np.array(query_vector).astype('float32')
        distances, indices = faiss_index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(documents):
                doc = documents[idx].copy()
                doc['distance'] = float(dist)
                if len(results) < 3 or dist < 9.0:
                    results.append(doc)
        return results
    except Exception as e:
        logger.error(f"Error búsqueda: {e}")
        return []

# ---------------------------
# DeepSeek async
# ---------------------------

async def call_deepseek_async(prompt, timeout=DEEPSEEK_TIMEOUT):
    """Llamada async a DeepSeek"""
    try:
        if not DEEPSEEK_API_KEY:
            logger.error("❌ DEEPSEEK_API_KEY faltante")
            return None
        
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 450,
            "top_p": 0.85
        }

        async with http_session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                data = await response.json()
                content = data['choices'][0]['message']['content'].strip()
                return content
            else:
                logger.error(f"DeepSeek error: {response.status}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout DeepSeek ({timeout}s)")
        return None
    except Exception as e:
        logger.error(f"Error DeepSeek: {e}")
        return None

# ---------------------------
# Response generation
# ---------------------------

async def generate_response_async(user_message, context="", history="", is_first_message=False):
    """Generar respuesta async"""
    
    if is_first_message:
        return (
            "¡Hola! 👋 Soy tu asistente virtual del *Vicerrectorado de Investigación* de la UNA Puno.\n\n"
            "📌 *Puedo ayudarte :*\n"
            "• información general sobre los procesos de proyecto y borrador de tesis de pregrado\n"
            "• Información de contacto de las coordinaciones de Investigación de \n"
            "• Horarios de atención\n"
            "• Ubicaciones de oficinas\n"

            "💡 *Comandos útiles:*\n"
            "/ayuda - Ver esta información\n"
            "/reset - Reiniciar conversación\n\n"
            "⏱️ Tu conversación estará activa por 30 minutos.\n"
            "🔒 Este chat es monitoreado para mejorar nuestro servicio.\n\n"
            "¿En qué puedo ayudarte hoy?"
        ), "welcome"
    
    system_prompt = r'''Eres el asistente virtual del Vicerrectorado de Investigación de la Universidad Nacional del Altiplano (UNA Puno).

TU PROPÓSITO:
Ayudar con información del Vicerrectorado de Investigación:
📧 Contactos (emails y teléfonos)
🕐 Horarios de atención  
📍 Ubicaciones
📚 Coordinación de investigación y tesis
   informacion general sobre los procesos y borrador de tesis
   preguntas frecuentes sobre los procesos de borrador de tesis

PERSONALIDAD:
- Profesional pero cercano
- Usas emojis estratégicamente (2-3 por mensaje)
- Claro, directo y útil
- Empático y servicial
- Información completa pero concisa

REGLAS:
- Máximo 120 palabras
- Usa información del contexto directamente
- NO inventes datos
- NO mezcles información de diferentes facultades
- Siempre ofrece más ayuda
- Sé específico con números, ubicaciones y horarios
- Si te hacen preguntas fuera de tu alcance (clima, matemáticas, chistes), redirige amablemente
- Recuerda que esta conversación es monitoreada para mejorar el servicio'''

    history_section = f"CONVERSACIÓN PREVIA:\n{history}\n\n" if history else ""
    
    if context:
        full_prompt = f"{system_prompt}\n\nINFORMACIÓN:\n{context}\n\n{history_section}PREGUNTA: {user_message}\n\nRESPUESTA:"
    else:
        full_prompt = f"{system_prompt}\n\n{history_section}PREGUNTA: {user_message}\n\nRESPUESTA:"

    response = await call_deepseek_async(full_prompt)
    
    if response:
        return response, DEEPSEEK_MODEL
    return "Lo siento, tengo problemas técnicos. Por favor, intenta de nuevo en unos momentos. 🔧", "error"

# ---------------------------
# Inactivity checker
# ---------------------------

async def check_inactive_users():
    """Verificar usuarios inactivos"""
    while True:
        try:
            await asyncio.sleep(60)
            
            current_time = datetime.now()
            inactive_users = []
            
            for phone, last_time in list(user_last_activity.items()):
                time_diff = (current_time - last_time).total_seconds()
                
                if time_diff >= INACTIVITY_TIMEOUT and phone not in user_closed_sessions:
                    inactive_users.append(phone)
            
            for phone in inactive_users:
                try:
                    closure_message = (
                        "Fue un placer ayudarte hoy 😊\n\n"
                        "Tu sesión ha expirado por inactividad, pero siempre puedes contactarme cuando necesites información sobre el Vicerrectorado de Investigación.\n\n"
                        "¡Que tengas un excelente día! 🌟\n\n"
                        "_Para iniciar una nueva conversación, simplemente envía un mensaje._"
                    )
                    
                    whatsapp_client.send_text(phone, closure_message)
                    user_closed_sessions.add(phone)
                    
                    asyncio.create_task(save_conversation_async(
                        phone, "[CIERRE_AUTOMATICO]", closure_message, "system", 0
                    ))
                    
                    logger.info(f"🔒 Sesión cerrada por inactividad: {phone}")
                    
                except Exception as e:
                    logger.error(f"Error cerrando sesión {phone}: {e}")
                    
        except Exception as e:
            logger.error(f"Error en check_inactive_users: {e}")

# ---------------------------
# Message processing async
# ---------------------------

async def process_message_async(user_message, phone_number):
    """Procesar mensaje async - RESPONDE RÁPIDO, GUARDA DESPUÉS"""
    async with semaphore:
        start_time = time.time()
        user_message = user_message.strip()
        
        user_last_activity[phone_number] = datetime.now()
        
        is_new_session = phone_number in user_closed_sessions
        if is_new_session:
            user_closed_sessions.remove(phone_number)

        if user_message.lower() == '/reset':
            user_closed_sessions.discard(phone_number)
            return "✓ Conversación reiniciada. ¿En qué puedo ayudarte?"

        if user_message.lower() in ['/ayuda', '/help', '/inicio', '/start']:
            response, _ = await generate_response_async("", "", "", is_first_message=True)
            return response

        trivial = ['hora', 'fecha', 'clima', 'chiste', 'fútbol', 'matemática']
        if any(k in user_message.lower() for k in trivial):
            if not any(w in user_message.lower() for w in ['universidad', 'facultad', 'correo']):
                return "Disculpa 😊, mi especialidad es información del Vicerrectorado de Investigación. ¿Puedo ayudarte con algún contacto, horario o ubicación?"

        if user_message.lower() in ['hola', 'hi', 'hello', 'buenos días', 'buenas tardes', 'buenas noches']:
            response, model = await generate_response_async("", "", "", is_first_message=True)
            asyncio.create_task(save_conversation_async(
                phone_number, user_message, response, model, int((time.time() - start_time) * 1000)
            ))
            return response

        loop = asyncio.get_event_loop()
        relevant_docs = await loop.run_in_executor(
            None, search_knowledge_base_cached, user_message[:200], 10
        )

        context = ""
        if relevant_docs:
            context = "\n\n".join([doc['content'][:1000] for doc in relevant_docs[:3]])

        history_task = asyncio.create_task(get_conversation_history_async(phone_number))
        history = await history_task
        
        response, model_used = await generate_response_async(user_message, context, history)

        if len(response) > 1600:
            response = response[:1597] + "..."

        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Guardar NO bloqueante
        asyncio.create_task(save_conversation_async(
            phone_number, user_message, response, model_used, response_time_ms
        ))

        logger.info(f"⚡ Respuesta ({model_used}, {response_time_ms}ms): {phone_number}")
        return response

# ---------------------------
# WhatsApp handler - CALLBACK para start_polling
# ---------------------------

def handle_incoming_message_sync(message):
    """Handler SYNC que WhatsAppAPIClient.start_polling() llama"""
    try:
        phone_number = extract_phone_number(message)
        user_message = message.get('body', '').strip()

        logger.info(f"📨 {phone_number}: {user_message[:50]}")

        # Ejecutar tarea async en el event loop
        future = asyncio.run_coroutine_threadsafe(
            process_and_send(phone_number, user_message),
            event_loop
        )
        
    except Exception as e:
        logger.error(f"Error handler: {e}", exc_info=True)

async def process_and_send(phone_number, user_message):
    """Procesar y enviar respuesta"""
    try:
        bot_response = await process_message_async(user_message, phone_number)
        success = whatsapp_client.send_text(phone_number, bot_response)
        
        if success:
            logger.info(f"✅ Enviado a {phone_number}")
        else:
            logger.error(f"❌ Error enviando a {phone_number}")
            
    except Exception as e:
        logger.error(f"Error en process_and_send: {e}", exc_info=True)

# ---------------------------
# Main async
# ---------------------------

async def main():
    global http_session, whatsapp_client, event_loop
    
    logger.info("=" * 60)
    logger.info("CHATBOT ASYNC - UNA PUNO")
    logger.info("✓ Alta concurrencia (150+ usuarios)")
    logger.info("✓ Cierre automático (30 min inactividad)")
    logger.info("✓ Guardado no bloqueante")
    logger.info("=" * 60)

    # Guardar referencia al event loop
    event_loop = asyncio.get_running_loop()

    http_session = aiohttp.ClientSession()

    logger.info("📊 PostgreSQL async...")
    if not await init_db_pool_async():
        logger.error("❌ PostgreSQL falló")
        return

    if not load_knowledge_base():
        logger.error("❌ KB falló")
        return

    logger.info("📱 WhatsApp API...")
    whatsapp_client = WhatsAppAPIClient(WHATSAPP_API_URL, WHATSAPP_API_KEY)
    
    if not whatsapp_client.check_connection():
        logger.error("❌ WhatsApp falló")
        return

    logger.info("✅ Todo listo")
    logger.info(f"🚀 Concurrencia máxima: {MAX_CONCURRENT}")
    logger.info(f"⏱️ Timeout inactividad: {INACTIVITY_TIMEOUT}s (30 min)")
    logger.info(f"🤖 Modelo: {DEEPSEEK_MODEL}")
    logger.info("=" * 60)

    # Iniciar task de verificación de inactividad
    asyncio.create_task(check_inactive_users())

    # Usar start_polling en un thread separado
    def run_polling():
        whatsapp_client.start_polling(handle_incoming_message_sync, interval=POLLING_INTERVAL)
    
    polling_thread = threading.Thread(target=run_polling, daemon=True)
    polling_thread.start()
    
    logger.info("🔄 Polling iniciado en thread separado")

    try:
        # Mantener el loop corriendo
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Deteniendo...")
    finally:
        await http_session.close()
        await db_pool.close()

if __name__ == '__main__':
    asyncio.run(main())