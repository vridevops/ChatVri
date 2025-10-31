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

# ===== CONFIGURACIÓN PARA 200 USUARIOS =====
MAX_CONCURRENT_MESSAGES = 50      # ← ANTES: 10, AHORA: 50
POLLING_INTERVAL = 3              # ← ANTES: 5, AHORA: 3 (más responsivo)
MAX_HISTORY_MESSAGES = 3          # ← Mantener en 3
INACTIVITY_TIMEOUT = 600        # ← 10 minutos
RATE_LIMIT_DELAY = 0.1            # ← NUEVO: 100ms entre mensajes
# Semáforo para controlar concurrencia
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

# Mapeo de facultades para búsqueda estricta
FACULTY_MAPPING = {
    'estadística': 'FACULTAD_DE_INGENIERIA_ESTADISTICA_E_INFORMATICA',
    'estadistica': 'FACULTAD_DE_INGENIERIA_ESTADISTICA_E_INFORMATICA', 
    'enfermería': 'FACULTAD_DE_ENFERMERIA',
    'enfermeria': 'FACULTAD_DE_ENFERMERIA',
    'agrarias': 'FACULTAD_DE_CIENCIAS_AGRARIAS',
    'veterinaria': 'FACULTAD_DE_MEDICINA_VETERINARIA_Y_ZOOTECNIA',
    'medicina veterinaria': 'FACULTAD_DE_MEDICINA_VETERINARIA_Y_ZOOTECNIA',
    'ingeniería económica': 'FACULTAD_DE_INGENIERIA_ECONOMICA',
    'ingenieria economica': 'FACULTAD_DE_INGENIERIA_ECONOMICA',
    'contables': 'FACULTAD_DE_CIENCIAS_CONTABLES_Y_ADMINISTRATIVAS',
    'administrativas': 'FACULTAD_DE_CIENCIAS_CONTABLES_Y_ADMINISTRATIVAS',
    'trabajo social': 'FACULTAD_DE_TRABAJO_SOCIAL',
    'ciencias sociales': 'FACULTAD_DE_CIENCIAS_SOCIALES',
    'minas': 'FACULTAD_DE_INGENIERIA_DE_MINAS',
    'derecho': 'FACULTAD_DE_CIENCIAS_JURIDICAS_Y_POLITICAS',
    'jurídicas': 'FACULTAD_DE_CIENCIAS_JURIDICAS_Y_POLITICAS',
    'química': 'FACULTAD_DE_INGENIERIA_QUIMICA',
    'quimica': 'FACULTAD_DE_INGENIERIA_QUIMICA',
    'biológicas': 'FACULTAD_DE_CIENCIAS_BIOLOGICAS',
    'biologicas': 'FACULTAD_DE_CIENCIAS_BIOLOGICAS',
    'educación': 'FACULTAD_DE_CIENCIAS_DE_LA_EDUCACION',
    'educacion': 'FACULTAD_DE_CIENCIAS_DE_LA_EDUCACION',
    'geológica': 'FACULTAD_DE_INGENIERIA_GEOLOGICA_Y_METALURGICA',
    'geologica': 'FACULTAD_DE_INGENIERIA_GEOLOGICA_Y_METALURGICA',
    'civil': 'FACULTAD_DE_INGENIERIA_CIVIL_Y_ARQUITECTURA',
    'arquitectura': 'FACULTAD_DE_INGENIERIA_CIVIL_Y_ARQUITECTURA',
    'agrícola': 'FACULTAD_DE_INGENIERIA_AGRICOLA',
    'agricola': 'FACULTAD_DE_INGENIERIA_AGRICOLA',
    'salud': 'FACULTAD_DE_CIENCIAS_DE_LA_SALUD',
    'mecánica': 'FACULTAD_DE_INGENIERIA_MECANICA_ELECTRICA_ELECTRONICA_Y_SISTEMAS',
    'mecanica': 'FACULTAD_DE_INGENIERIA_MECANICA_ELECTRICA_ELECTRONICA_Y_SISTEMAS',
    'medicina humana': 'FACULTAD_DE_MEDICINA_HUMANA',
    'administrativas humanas': 'FACULTAD_DE_CIENCIAS_ADMINISTRATIVAS_Y_HUMANAS',
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
        
        logger.info(f"✓ Base cargada: {len(documents)} docs")
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
    """Caché de búsquedas para mejorar rendimiento"""
    return strict_search_knowledge_base(query, top_k)

def strict_search_knowledge_base(query, top_k=3, similarity_threshold=0.75):
    """Búsqueda estricta que prioriza matching exacto de facultades"""
    if not embedding_model or not faiss_index:
        return []
    
    try:
        # Normalizar query para matching de facultades
        query_lower = query.lower()
        
        # Buscar facultad específica en la query
        target_faculty = None
        for term, faculty in FACULTY_MAPPING.items():
            if term in query_lower:
                target_faculty = faculty
                break
        
        # Búsqueda semántica normal
        query_vector = embedding_model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        distances, indices = faiss_index.search(query_vector, top_k * 5)  # Buscar más resultados
        
        # Filtrar estrictamente
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(documents):
                continue
                
            doc = documents[idx].copy()
            doc['similarity'] = 1 / (1 + dist)
            
            # ⭐ FILTRO ESTRICTO: Solo documentos con alta similitud
            if doc['similarity'] < similarity_threshold:
                continue
            
            # ⭐ FILTRO POR FACULTAD: Si se busca una facultad específica
            if target_faculty:
                doc_facultad = doc.get('facultad', '').upper()
                if target_faculty in doc_facultad:
                    doc['faculty_match'] = True
                    results.append(doc)
                else:
                    continue
            else:
                results.append(doc)
        
        # Ordenar por similitud y facultad match
        results.sort(key=lambda x: (x.get('faculty_match', False), x['similarity']), reverse=True)
        
        logger.info(f"🔍 Búsqueda: '{query}' -> {len(results)} resultados (umbral: {similarity_threshold})")
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Error búsqueda estricta: {e}")
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
            "temperature": 0.3,  # ⭐ REDUCIDO para menos creatividad
            "max_tokens": 400,
            "top_p": 0.7
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
# Response generation
# ---------------------------

STRICT_SYSTEM_PROMPT = r'''Eres asistente virtual del Vicerrectorado de Investigación UNA Puno.

🔒 **REGLAS ESTRICTAS - CRÍTICO:**
1. SOLO usa la información del CONTEXTO proporcionado
2. NUNCA inventes datos, contactos, horarios o líneas de investigación
3. Si la información no está en el CONTEXTO, di exactamente: "No tengo información específica sobre eso en mi base de datos"
4. NO combines información de diferentes facultades
5. NO extrapoles o deduzcas información no presente
6. NO crees emails, teléfonos o ubicaciones que no existen
7. Si no hay información suficiente, sé honesto y di que no tienes los datos

🎯 **INFORMACIÓN QUE MANEJAS:**
- Coordinadores por facultad (nombres, emails, teléfonos EXACTOS)
- Líneas y sublíneas de investigación POR FACULTAD Y ESCUELA
- Procesos de tesis según reglamento
- Preguntas frecuentes

📋 **CUANDO NO HAY INFORMACIÓN:**
Responde EXACTAMENTE: "No encuentro información específica sobre eso en mi base de conocimiento. Te recomiendo contactar directamente al Vicerrectorado de Investigación."

❌ **PROHIBIDO ABSOLUTO:**
- Crear emails que no existen
- Inventar teléfonos o horarios
- Generar líneas de investigación no listadas
- Modificar ubicaciones o contactos
- Usar conocimiento externo o información no verificada

PERSONALIDAD:
- Profesional pero cercano
- Usa emojis estratégicamente (1-2 máximo)
- Claro, directo y útil
- Honesto sobre limitaciones

CONTEXTO DISPONIBLE:
{context}

PREGUNTA: {user_query}

RESPUESTA (SOLO con información del contexto, sé honesto si no hay datos):'''

async def verify_context_usage(user_query, bot_response, context_docs):
    """Verificar que la respuesta use SOLO información del contexto"""
    if not context_docs:
        return "No tengo información específica sobre eso en mi base de datos. Te recomiendo contactar directamente al Vicerrectorado de Investigación."
    
    # Verificación simple: si no hay docs relevantes, no debería dar información específica
    if not context_docs and any(keyword in bot_response.lower() for keyword in ['email', 'teléfono', 'horario', 'ubicación', 'línea', 'investigación']):
        return "No tengo información específica sobre eso en mi base de datos. Te recomiendo contactar directamente al Vicerrectorado de Investigación."
    
    return bot_response

async def generate_response_async(user_message, context_docs=[], history="", is_first_message=False):
    """Generar respuesta con verificaciones estrictas"""
    
    if is_first_message:
        return (
            "¡Hola! 👋 Soy tu asistente virtual del *Vicerrectorado de Investigación* de la UNA Puno.\n\n"
            "📌 *Puedo ayudarte con:*\n"
            "• Información general sobre procesos de proyecto y borrador de tesis\n"
            "• Contactos de coordinaciones de investigación\n"
            "• Horarios de atención\n"
            "• Ubicaciones de oficinas\n"
            "• Líneas de investigación\n"
            "• Migración de cuenta a PGI\n\n"
            "💡 *Comandos útiles:*\n"
            "/ayuda - Ver esta información\n"
            "/reset - Reiniciar conversación\n\n"
            "⏱️ Tu conversación estará activa por 10 minutos.\n"
            "🔒 Este chat es monitoreado para mejorar nuestro servicio.\n\n"
            "¿En qué puedo ayudarte hoy?"
        ), "welcome"
    
    # Si no hay contexto relevante
    if not context_docs:
        return "No tengo información específica sobre ese tema en mi base de datos. Te recomiendo contactar directamente al Vicerrectorado de Investigación.", "no_context"
    
    # Construir contexto desde los documentos
    context_text = ""
    if context_docs:
        context_parts = []
        for doc in context_docs[:3]:  # Máximo 3 documentos
            context_parts.append(doc.get('text', '')[:800])
        context_text = "\n\n---\n\n".join(context_parts)
    
    history_section = f"CONVERSACIÓN PREVIA:\n{history}\n\n" if history else ""
    
    # Usar prompt estricto
    full_prompt = STRICT_SYSTEM_PROMPT.format(
        context=context_text,
        user_query=user_message
    )
    
    response = await call_deepseek_async(full_prompt)
    
    if response:
        # Verificación estricta
        verified_response = await verify_context_usage(user_message, response, context_docs)
        return verified_response, DEEPSEEK_MODEL
    
    return "Disculpa, tengo dificultades técnicas. Por favor intenta nuevamente.", "error"

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
            if not any(w in user_message.lower() for w in ['universidad', 'facultad', 'correo', 'tesis', 'investigación']):
                return "Disculpa 😊, mi especialidad es información del Vicerrectorado de Investigación. ¿Puedo ayudarte con algún contacto, horario, ubicación o proceso de tesis?"

        if user_message.lower() in ['hola', 'hi', 'hello', 'buenos días', 'buenas tardes', 'buenas noches']:
            response, model = await generate_response_async("", [], "", is_first_message=True)
            asyncio.create_task(save_conversation_async(
                phone_number, user_message, response, model, int((time.time() - start_time) * 1000)
            ))
            return response

        # ⭐ BÚSQUEDA ESTRICTA MEJORADA
        loop = asyncio.get_event_loop()
        relevant_docs = await loop.run_in_executor(
            None, strict_search_knowledge_base, user_message, 3, 0.7
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
    logger.info("CHATBOT ASYNC - UNA PUNO - VERSIÓN ESTRICTA")
    logger.info("✅ Búsqueda estricta por facultades")
    logger.info("✅ Verificación de contexto")
    logger.info("✅ Sin invención de datos")
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
    logger.info(f"🤖 Modelo: {DEEPSEEK_MODEL} (temp: 0.3)")
    logger.info(f"🔍 Umbral similitud: 0.7")
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