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
import re

from whatsapp_client import WhatsAppAPIClient, extract_phone_number

load_dotenv()
# Configuración
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIGURACIÓN MEJORADA =====
MAX_CONCURRENT_MESSAGES = 50
POLLING_INTERVAL = 3
MAX_HISTORY_MESSAGES = 3
INACTIVITY_TIMEOUT = 600
RATE_LIMIT_DELAY = 0.1
semaphore = asyncio.Semaphore(MAX_CONCURRENT_MESSAGES)

# Cache de conversaciones en memoria
conversation_cache = {}

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
# Term expansion mejorado
# ---------------------------
TERM_EXPANSION = {
    'email': ['correo', 'mail', 'correo electrónico'],
    'correo': ['email', 'mail'],
    'teléfono': ['celular', 'telefono', 'número', 'contacto'],
    'celular': ['teléfono', 'telefono', 'número'],
    'horario': ['hora', 'horarios', 'atención'],
    'ubicación': ['ubicacion', 'lugar', 'donde', 'dirección'],
    'línea': ['linea', 'investigación', 'investigacion'],
    'enfermería': ['enfermeria', 'enfermera'],
    'estadística': ['estadistica', 'estadisticas'],
}

# Mapeo de facultades para búsqueda optimizada
FACULTY_EXPANSIONS = {
    'estadística': ['estadistica', 'estadisticas', 'fie', 'estadística e informática'],
    'estadistica': ['estadística', 'estadisticas', 'fie', 'estadística e informática'],
    'agrarias': ['ciencias agrarias', 'fca', 'agronomía', 'agronomia'],
    'enfermería': ['enfermeria', 'enfermera', 'enfermero'],
    'veterinaria': ['medicina veterinaria', 'fmvz', 'zootecnia'],
    'contables': ['ciencias contables', 'fcca', 'contabilidad'],
    'económica': ['ingeniería económica', 'fie', 'economica'],
    'minas': ['ingeniería de minas', 'fim'],
    'derecho': ['ciencias jurídicas', 'fcjp'],
    'química': ['ingeniería química', 'fiq'],
    'biológicas': ['ciencias biológicas', 'fcb'],
    'sociales': ['ciencias sociales', 'fcs'],
    'educación': ['ciencias de la educación', 'fceduc'],
    'geológica': ['ingeniería geológica', 'figmm'],
    'civil': ['ingeniería civil', 'fica'],
    'agrícola': ['ingeniería agrícola', 'fia'],
    'salud': ['ciencias de la salud'],
    'mecánica': ['ingeniería mecánica', 'fimees'],
    'medicina': ['medicina humana', 'fmh'],
    'administrativas': ['ciencias administrativas', 'fcah'],
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
# Knowledge base - BÚSQUEDA OPTIMIZADA
# ---------------------------

def load_knowledge_base(index_path='faiss_index.bin', json_path='knowledge_base.json'):
    """Cargar base de conocimiento FAISS + documentos JSON"""
    global embedding_model, faiss_index, documents
    try:
        logger.info("Cargando embeddings...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        logger.info(f"Cargando índice FAISS desde {index_path}...")
        faiss_index = faiss.read_index(index_path)
        
        logger.info(f"Cargando documentos desde {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
            documents = knowledge_data.get('documents', knowledge_data)
        
        # Log de estadísticas de la base de conocimiento
        type_counts = {}
        for doc in documents:
            doc_type = doc.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        logger.info(f"✅ Base cargada: {len(documents)} documentos")
        logger.info(f"📊 Distribución: {type_counts}")
        
        return True
    except Exception as e:
        logger.error(f"Error KB: {e}")
        return False

def expand_query(query):
    """Expandir términos de búsqueda con sinónimos"""
    expanded_terms = [query.lower()]
    for word in query.lower().split():
        if word in TERM_EXPANSION:
            expanded_terms.extend(TERM_EXPANSION[word])
    return ' '.join(expanded_terms)

@lru_cache(maxsize=500)
def search_knowledge_base_cached(query, top_k=5):
    """Caché de búsquedas optimizadas"""
    return optimized_search_knowledge_base(query, top_k, similarity_threshold=0.4)

def optimized_search_knowledge_base(query, top_k=5, similarity_threshold=0.4):
    """Búsqueda optimizada con mejor matching de facultades"""
    if not embedding_model or not faiss_index:
        return []
    
    try:
        query_lower = query.lower()
        logger.info(f"🔍 Búsqueda optimizada: '{query}'")
        
        # ⭐ EXPANSIÓN MEJORADA DE TÉRMINOS
        expanded_terms = [query_lower]
        
        # Expansión específica por facultad
        for term, expansions in FACULTY_EXPANSIONS.items():
            if term in query_lower:
                expanded_terms.extend(expansions)
        
        # Búsqueda semántica principal
        all_results = []
        for term in set(expanded_terms):  # Eliminar duplicados
            query_vector = embedding_model.encode([term])
            query_vector = np.array(query_vector).astype('float32')
            
            distances, indices = faiss_index.search(query_vector, top_k * 3)
            similarities = 1 / (1 + distances[0])
            
            for i, (similarity, idx) in enumerate(zip(similarities, indices[0])):
                if idx >= len(documents):
                    continue
                    
                doc = documents[idx].copy()
                doc['similarity'] = float(similarity)
                
                # ⭐ SCORING MEJORADO
                score = similarity
                
                # Bonus por matching exacto de facultad
                doc_facultad = doc.get('facultad', '').lower()
                doc_text = doc.get('text', '').lower()
                
                # Bonus si la query menciona una facultad y el documento es de esa facultad
                for faculty_keyword in FACULTY_EXPANSIONS.keys():
                    if faculty_keyword in query_lower and faculty_keyword in doc_facultad:
                        score += 0.3
                        break
                
                # Bonus por tipo de documento relevante
                doc_type = doc.get('type', '')
                if 'contacto' in query_lower and 'coordinador' in doc_type:
                    score += 0.2
                if any(word in query_lower for word in ['línea', 'linea', 'investigación']) and 'linea_investigacion' in doc_type:
                    score += 0.2
                
                # Bonus por matching de palabras clave en el texto
                query_words = set(query_lower.split())
                doc_words = set(doc_text.split())
                keyword_matches = len(query_words.intersection(doc_words))
                score += (keyword_matches * 0.05)
                
                doc['combined_score'] = score
                
                if score >= similarity_threshold:
                    all_results.append(doc)
        
        # Eliminar duplicados y ordenar
        unique_results = []
        seen_doc_ids = set()
        
        for result in all_results:
            # Usar combinación de texto y facultad para identificar duplicados
            doc_id = f"{result.get('text','')[:50]}_{result.get('facultad','')}"
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_results.append(result)
        
        # Ordenar por score combinado
        unique_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"📊 Resultados para '{query}': {len(unique_results)} (umbral: {similarity_threshold})")
        
        # Log de top resultados para debugging
        for i, result in enumerate(unique_results[:3]):
            logger.info(f"   Top {i+1}: score={result['combined_score']:.3f}, tipo={result.get('type','?')}, fac={result.get('facultad','?')}")
        
        return unique_results[:top_k]
        
    except Exception as e:
        logger.error(f"❌ Error en búsqueda optimizada: {e}")
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
            "temperature": 0.4,
            "max_tokens": 500,
            "top_p": 0.8
        }

        async with http_session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                data = await response.json()
                content = data['choices'][0]['message']['content'].strip()
                return content
            else:
                error_text = await response.text()
                logger.error(f"DeepSeek error: {response.status} - {error_text}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout DeepSeek ({timeout}s)")
        return None
    except Exception as e:
        logger.error(f"Error DeepSeek: {e}")
        return None

# ---------------------------
# Response generation - PROMPT MEJORADO
# ---------------------------

IMPROVED_SYSTEM_PROMPT = r'''Eres asistente virtual del Vicerrectorado de Investigación UNA Puno.

🎯 **INFORMACIÓN QUE MANEJAS:**
- Coordinadores por facultad (contactos exactos)
- Líneas y sublíneas de investigación  
- Procesos de tesis y reglamentos
- Preguntas frecuentes

🔍 **INSTRUCCIONES ESPECÍFICAS:**
1. Cuando el usuario pregunte por una facultad, proporciona TODA la información relevante del contexto
2. Si hay información de contacto Y líneas de investigación, incluye ambas
3. Sé específico con los datos: nombres exactos, emails, teléfonos, líneas de investigación
4. Usa emojis relevantes para hacer la información más clara
5. Si el contexto tiene la información, NO digas que no la tienes

📝 **FORMATO DE RESPUESTAS:**
- Para coordinadores: 👨‍💼 Coordinador, 📧 Email, 📱 Teléfono, 🕐 Horario, 📍 Ubicación
- Para líneas de investigación: 🔬 Línea, 📚 Sublíneas
- Combina información cuando sea relevante

❌ **NO DIGAS:**
- "No tengo información específica" (si el contexto tiene datos)
- "Contacta al Vicerrectorado" (como primera opción)
- Información genérica sin datos concretos

CONTEXTO DISPONIBLE:
{context}

PREGUNTA: {user_query}

Proporciona una respuesta COMPLETA con toda la información relevante del contexto:'''

async def generate_response_async(user_message, context_docs=[], history="", is_first_message=False):
    """Generar respuesta con contexto mejorado"""
    
    if is_first_message:
        return (
            "¡Hola! 👋 Soy tu asistente virtual del *Vicerrectorado de Investigación* de la UNA Puno.\n\n"
            "📌 *Puedo ayudarte con:*\n"
            "• Información de coordinadores por facultad\n"
            "• Líneas y sublíneas de investigación\n"
            "• Procesos de tesis y reglamentos\n"
            "• Preguntas frecuentes sobre PGI\n\n"
            "💡 *Comandos útiles:*\n"
            "/ayuda - Ver esta información\n"
            "/reset - Reiniciar conversación\n\n"
            "⏱️ Sesión activa por 10 minutos\n"
            "🔒 Chat monitoreado para mejora continua\n\n"
            "¿En qué puedo ayudarte?"
        ), "welcome"
    
    # Si no hay contexto relevante
    if not context_docs:
        return "No encuentro información específica sobre ese tema en mi base de conocimiento actual. Te recomiendo contactar directamente con la coordinación de investigación de la facultad correspondiente.", "no_context"
    
    # ⭐ CONSTRUIR CONTEXTO MEJORADO
    context_text = ""
    if context_docs:
        context_parts = []
        for doc in context_docs[:4]:  # Más documentos para mejor contexto
            # Enriquecer con información de tipo y facultad
            doc_type = doc.get('type', '')
            doc_facultad = doc.get('facultad', '')
            
            header = ""
            if doc_facultad:
                header += f"[Facultad: {doc_facultad}]"
            if doc_type:
                header += f"[Tipo: {doc_type}]"
            
            doc_content = doc.get('text', '')
            if header:
                context_parts.append(f"{header}\n{doc_content}")
            else:
                context_parts.append(doc_content)
        
        context_text = "\n\n---\n\n".join(context_parts)
        logger.info(f"📄 Contexto construido: {len(context_docs)} docs, {len(context_text)} chars")
    
    history_section = f"CONVERSACIÓN PREVIA:\n{history}\n\n" if history else ""
    
    # Usar prompt mejorado
    full_prompt = IMPROVED_SYSTEM_PROMPT.format(
        context=context_text,
        user_query=user_message
    )
    
    response = await call_deepseek_async(full_prompt)
    
    if response:
        return response, DEEPSEEK_MODEL
    
    return "Disculpa, tengo dificultades técnicas en este momento. Por favor intenta nuevamente o contacta directamente al Vicerrectorado de Investigación.", "error"

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
# Message processing async - OPTIMIZADO
# ---------------------------

async def process_message_async(user_message, phone_number):
    """Procesar mensaje con búsqueda optimizada"""
    async with semaphore:
        start_time = time.time()
        user_message = user_message.strip()
        
        user_last_activity[phone_number] = datetime.now()
        
        is_new_session = phone_number in user_closed_sessions
        if is_new_session:
            user_closed_sessions.remove(phone_number)

        # Comandos especiales
        if user_message.lower() == '/reset':
            user_closed_sessions.discard(phone_number)
            return "✓ Conversación reiniciada. ¿En qué puedo ayudarte?"

        if user_message.lower() in ['/ayuda', '/help', '/inicio', '/start']:
            response, _ = await generate_response_async("", [], "", is_first_message=True)
            return response

        # Filtrar preguntas fuera de contexto
        trivial = ['hora', 'fecha', 'clima', 'chiste', 'fútbol', 'matemática', 'programación']
        if any(k in user_message.lower() for k in trivial):
            if not any(w in user_message.lower() for w in ['universidad', 'facultad', 'correo', 'tesis', 'investigación', 'linea']):
                return "Disculpa 😊, mi especialidad es información del Vicerrectorado de Investigación. ¿Puedo ayudarte con contactos, líneas de investigación o procesos de tesis?"

        if user_message.lower() in ['hola', 'hi', 'hello', 'buenos días', 'buenas tardes', 'buenas noches']:
            response, model = await generate_response_async("", [], "", is_first_message=True)
            asyncio.create_task(save_conversation_async(
                phone_number, user_message, response, model, int((time.time() - start_time) * 1000)
            ))
            return response

        # ⭐ BÚSQUEDA OPTIMIZADA
        loop = asyncio.get_event_loop()
        relevant_docs = await loop.run_in_executor(
            None, optimized_search_knowledge_base, user_message, 5, 0.4
        )

        # Obtener historial de conversación
        history_task = asyncio.create_task(get_conversation_history_async(phone_number))
        history = await history_task
        
        # Generar respuesta con documentos específicos
        response, model_used = await generate_response_async(user_message, relevant_docs, history)

        # Limitar longitud para WhatsApp
        if len(response) > 1600:
            response = response[:1597] + "..."

        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Guardar NO bloqueante
        asyncio.create_task(save_conversation_async(
            phone_number, user_message, response, model_used, response_time_ms
        ))

        logger.info(f"⚡ Respuesta ({model_used}, {response_time_ms}ms, docs: {len(relevant_docs)}): {phone_number}")
        return response

# ---------------------------
# WhatsApp handler
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
        
        # Opcional: esperar un poco para rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
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
    logger.info("CHATBOT UNA PUNO - VERSIÓN OPTIMIZADA")
    logger.info("✅ Búsqueda optimizada por facultades")
    logger.info("✅ Mejor matching semántico")
    logger.info("✅ Respuestas específicas con datos concretos")
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
    logger.info(f"⏱️ Timeout inactividad: {INACTIVITY_TIMEOUT}s")
    logger.info(f"🤖 Modelo: {DEEPSEEK_MODEL} (temp: 0.4)")
    logger.info(f"🔍 Umbral similitud: 0.4")
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
        if db_pool:
            await db_pool.close()

if __name__ == '__main__':
    asyncio.run(main())