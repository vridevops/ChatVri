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

# ===== CONFIGURACIÓN =====
MAX_CONCURRENT_MESSAGES = 50
POLLING_INTERVAL = 3
MAX_HISTORY_MESSAGES = 3
INACTIVITY_TIMEOUT = 600
RATE_LIMIT_DELAY = 0.1

# Cache de conversaciones en memoria
conversation_cache = {}

# Environment
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
DEEPSEEK_TIMEOUT = int(os.getenv('DEEPSEEK_TIMEOUT', '20'))

WHATSAPP_API_URL = os.getenv('WHATSAPP_API_URL')
WHATSAPP_API_KEY = os.getenv('WHATSAPP_API_KEY')

# ⭐ NUEVO: File Server
FILE_SERVER_URL = os.getenv('FILE_SERVER_URL', 'https://files.services.vridevops.space')
FORMATOS_ENABLED = os.getenv('FORMATOS_ENABLED', 'true').lower() == 'true'

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

# Term expansion
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

FACULTY_EXPANSIONS = {
    'estadística': ['estadistica', 'estadisticas', 'FINESI', 'estadística e informática'],
    'estadistica': ['estadística', 'estadisticas', 'FINESI', 'estadística e informática'],
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

# ============================================================================
# DATABASE ASYNC
# ============================================================================

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



# ============================================================================
# BÚSQUEDA Y ENVÍO DE FORMATOS ⭐ NUEVO
# ============================================================================

async def buscar_y_enviar_formato(mensaje: str, phone_number: str) -> tuple[bool, bool]:
    """
    Detectar si el usuario pide un formato y enviarlo
    Maneja ambigüedad cuando hay múltiples escuelas
    
    Returns:
        (es_busqueda_formato, se_envio_exitosamente)
    """
    try:
        mensaje_lower = mensaje.lower()
        
        # Palabras clave que indican búsqueda de formato
        keywords_formato = ['formato', 'borrador', 'proyecto', 'plantilla', 'esquema']
        
        # Verificar si menciona formato
        if not any(kw in mensaje_lower for kw in keywords_formato):
            return (False, False)
        
        logger.info(f"🔍 Detectada solicitud de formato: '{mensaje}'")
        
        # Detectar tipo de formato
        tipo = None
        if 'borrador' in mensaje_lower:
            tipo = 'borrador'
        elif 'proyecto' in mensaje_lower:
            tipo = 'proyecto'
        
        # Extraer palabras clave - MEJORADO para mantener palabras importantes
        stop_words = {
            'dame', 'el', 'de', 'formato', 'tesis', 'necesito', 
            'quiero', 'para', 'mi', 'un', 'una', 'favor', 'por'
        }
        # NO eliminar: la, los, las (importantes para facultades)
        
        palabras = mensaje_lower.split()
        query_words = [p for p in palabras if p not in stop_words]
        
        if not query_words:
            logger.warning("No se encontraron palabras clave específicas")
            await whatsapp_client.send_text_async(
                phone_number,
                "⚠️ No entendí qué formato necesitas.\n\n"
                "💡 Especifica la facultad o carrera:\n"
                "• 'formato de proyecto de estadística'\n"
                "• 'borrador de turismo'\n"
                "• 'proyecto de educación inicial'"
            )
            return (True, False)
        
        query = ' '.join(query_words)
        logger.info(f"   Query extraído: '{query}' (tipo: {tipo or 'cualquiera'})")
        
        # BÚSQUEDA MEJORADA: Primero buscar todos los matches
        async with db_pool.acquire() as conn:
            # Buscar todos los formatos que coincidan
            formatos = await conn.fetch(
                """
                SELECT * FROM formatos_tesis
                WHERE activo = true
                AND ($2::text IS NULL OR tipo = $2)
                AND (
                    LOWER(facultad) LIKE '%' || LOWER($1) || '%'
                    OR LOWER(escuela_profesional) LIKE '%' || LOWER($1) || '%'
                    OR EXISTS (
                        SELECT 1 FROM unnest(keywords) kw 
                        WHERE LOWER(kw) LIKE '%' || LOWER($1) || '%'
                    )
                )
                ORDER BY 
                    CASE 
                        WHEN escuela_profesional IS NOT NULL THEN 1
                        ELSE 2
                    END,
                    CASE
                        WHEN LOWER(escuela_profesional) LIKE '%' || LOWER($1) || '%' THEN 1
                        WHEN LOWER(facultad) LIKE '%' || LOWER($1) || '%' THEN 2
                        ELSE 3
                    END
                """,
                query, tipo
            )
        
        if not formatos:
            # No se encontró ningún formato
            logger.warning(f"   ❌ Formato no encontrado para: '{query}'")
            
            # Buscar formatos similares para sugerencias
            async with db_pool.acquire() as conn:
                sugerencias = await conn.fetch(
                    """
                    SELECT DISTINCT
                        COALESCE(escuela_profesional, facultad) as nombre,
                        tipo
                    FROM formatos_tesis
                    WHERE activo = true
                    LIMIT 5
                    """
                )
            
            mensaje_sugerencias = "❌ No encontré el formato que buscas.\n\n💡 Formatos disponibles:\n"
            for sug in sugerencias[:3]:
                mensaje_sugerencias += f"• '{sug['tipo']} de {sug['nombre'].lower()}'\n"
            
            await whatsapp_client.send_text_async(phone_number, mensaje_sugerencias)
            return (True, False)
        
        # Si hay múltiples resultados de la MISMA facultad pero diferentes escuelas
        if len(formatos) > 1:
            facultad_principal = formatos[0]['facultad']
            todas_misma_facultad = all(f['facultad'] == facultad_principal for f in formatos)
            
            if todas_misma_facultad and any(f['escuela_profesional'] for f in formatos):
                # Hay múltiples escuelas profesionales, pedir aclaración
                logger.info(f"   ⚠️ Múltiples escuelas encontradas en {facultad_principal}")
                
                mensaje = f"📚 La facultad de *{facultad_principal}* tiene varias escuelas profesionales.\n\n"
                mensaje += "¿A cuál te refieres?\n\n"
                
                escuelas_unicas = set()
                for f in formatos:
                    if f['escuela_profesional']:
                        escuelas_unicas.add(f['escuela_profesional'])
                
                for escuela in sorted(escuelas_unicas):
                    mensaje += f"• {escuela}\n"
                
                mensaje += f"\n💡 Ejemplo: 'formato de proyecto de educación inicial'"
                
                await whatsapp_client.send_text_async(phone_number, mensaje)
                return (True, False)
        
        # Tomar el primer formato (más específico según ORDER BY)
        formato = formatos[0]
        logger.info(f"   ✅ Formato encontrado: {formato['codigo']}")
        
        # Crear URL temporal en File Server
        try:
            async with http_session.post(
                f"{FILE_SERVER_URL}/api/temp-url/{formato['id']}"
            ) as resp:
                if resp.status != 200:
                    logger.error(f"   ❌ Error File Server: {resp.status}")
                    await whatsapp_client.send_text_async(
                        phone_number,
                        "❌ Error generando el link de descarga. Intenta de nuevo."
                    )
                    return (True, False)
                
                data = await resp.json()
                download_url = data['url']
                logger.info(f"   📎 URL temporal creada: {download_url[:50]}...")
        
        except Exception as e:
            logger.error(f"   ❌ Error conectando File Server: {e}")
            await whatsapp_client.send_text_async(
                phone_number,
                "❌ Error generando el link de descarga. Intenta de nuevo."
            )
            return (True, False)
        
        # Construir mensaje
        escuela = formato['escuela_profesional']
        facultad = formato['facultad']
        
        if escuela:
            lugar = f"{escuela}\n📚 {facultad}"
        else:
            lugar = facultad
        
        caption = (
            f"📄 {formato['titulo']}\n\n"
            f"📍 {lugar}\n"
            f"📝 Tipo: {formato['tipo'].title()}\n"
            f"📦 Tamaño: {formato['file_size_kb']} KB\n\n"
            f"⚡ Link válido por 5 minutos"
        )
        
        # Enviar archivo por WhatsApp
        success = await whatsapp_client.send_media_async(
            to=phone_number,
            media_url=download_url,
            caption=caption
        )
        
        if success:
            # Registrar envío en base de datos
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "SELECT registrar_envio($1, $2, $3)",
                    formato['id'], phone_number, mensaje
                )
            
            logger.info(f"✅ Formato enviado: {formato['codigo']} → {phone_number}")
            return (True, True)
        else:
            logger.error(f"   ❌ Error enviando archivo por WhatsApp")
            await whatsapp_client.send_text_async(
                phone_number,
                "❌ Hubo un error al enviar el formato. Por favor intenta de nuevo."
            )
            return (True, False)
        
    except Exception as e:
        logger.error(f"Error en buscar_y_enviar_formato: {e}", exc_info=True)
        return (False, False)
    
# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

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


def optimized_search_knowledge_base(query, top_k=5, similarity_threshold=0.3):
    """Búsqueda optimizada con mejor matching"""
    if not embedding_model or not faiss_index:
        return []
    
    try:
        query_lower = query.lower()
        logger.info(f"🔍 Búsqueda optimizada: '{query}'")
        
        expanded_terms = [query_lower]
        
        FACULTY_DIRECT_MAPPING = {
            'enfermería': 'FACULTAD_DE_ENFERMERIA',
            'enfermeria': 'FACULTAD_DE_ENFERMERIA',
            'estadística': 'FACULTAD_DE_INGENIERIA_ESTADISTICA_E_INFORMATICA', 
            'estadistica': 'FACULTAD_DE_INGENIERIA_ESTADISTICA_E_INFORMATICA',
            'agrarias': 'FACULTAD_DE_CIENCIAS_AGRARIAS',
            'veterinaria': 'FACULTAD_DE_MEDICINA_VETERINARIA_Y_ZOOTECNIA',
            'contables': 'FACULTAD_DE_CIENCIAS_CONTABLES_Y_ADMINISTRATIVAS',
            'económica': 'FACULTAD_DE_INGENIERIA_ECONOMICA',
            'economica': 'FACULTAD_DE_INGENIERIA_ECONOMICA',
            'minas': 'FACULTAD_DE_INGENIERIA_DE_MINAS',
            'derecho': 'FACULTAD_DE_DERECHO_Y_CIENCIAS_POLITICAS',
            'civil': 'FACULTAD_DE_INGENIERIA_CIVIL',
            'medicina': 'FACULTAD_DE_MEDICINA_HUMANA'
        }
        
        for term, facultad_nombre in FACULTY_DIRECT_MAPPING.items():
            if term in query_lower:
                expanded_terms.append(facultad_nombre.lower())
                expanded_terms.append(term)
        
        all_results = []
        for term in set(expanded_terms):
            query_vector = embedding_model.encode([term])
            query_vector = np.array(query_vector).astype('float32')
            
            distances, indices = faiss_index.search(query_vector, top_k * 3)
            similarities = 1 / (1 + distances[0])
            
            for i, (similarity, idx) in enumerate(zip(similarities, indices[0])):
                if idx >= len(documents):
                    continue
                    
                doc = documents[idx].copy()
                doc['similarity'] = float(similarity)
                
                score = similarity
                doc_text = doc.get('text', '').lower()
                doc_facultad = doc.get('facultad', '').lower()
                
                for search_term, facultad_target in FACULTY_DIRECT_MAPPING.items():
                    if (search_term in query_lower and 
                        facultad_target.lower() in doc_facultad):
                        score += 0.5
                        logger.info(f"   🎯 MATCH EXACTO: '{search_term}' -> '{facultad_target}'")
                        break
                
                doc_type = doc.get('type', '')
                if any(word in query_lower for word in ['línea', 'linea', 'investigación', 'investigacion', 'sublinea']):
                    if 'linea_investigacion' in doc_type:
                        score += 0.3
                
                query_words = set(query_lower.split())
                doc_words = set(doc_text.split())
                keyword_matches = len(query_words.intersection(doc_words))
                score += (keyword_matches * 0.1)
                
                doc['combined_score'] = score
                
                if score >= similarity_threshold:
                    all_results.append(doc)
        
        unique_results = []
        seen_doc_ids = set()
        
        for result in all_results:
            doc_id = f"{result.get('text','')[:100]}_{result.get('facultad','')}"
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"📊 Resultados para '{query}': {len(unique_results)} documentos")
        for i, result in enumerate(unique_results[:3]):
            logger.info(f"   Top {i+1}: score={result['combined_score']:.3f}, " +
                       f"tipo={result.get('type','?')}, " +
                       f"facultad={result.get('facultad','?')}")
        
        return unique_results[:top_k]
        
    except Exception as e:
        logger.error(f"❌ Error en búsqueda optimizada: {e}")
        return []


def direct_faculty_search(query, docs, top_k=3):
    """Búsqueda directa por nombre de facultad - como fallback"""
    query_lower = query.lower()
    
    faculty_keywords = {
        'enfermería': 'enfermeria',
        'enfermeria': 'enfermeria', 
        'estadística': 'estadistica',
        'estadistica': 'estadistica',
        'agrarias': 'agrarias',
        'veterinaria': 'veterinaria',
        'contables': 'contables',
        'económica': 'economica',
        'economica': 'economica'
    }
    
    results = []
    for doc in docs:
        doc_facultad = doc.get('facultad', '').lower()
        doc_type = doc.get('type', '')
        doc_text = doc.get('text', '').lower()
        
        for keyword, faculty_type in faculty_keywords.items():
            if (keyword in query_lower and 
                faculty_type in doc_facultad and
                'linea_investigacion' in doc_type):
                results.append(doc)
                break
        
        if len(results) >= top_k:
            break
    
    return results


# ============================================================================
# DEEPSEEK
# ============================================================================

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


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

IMPROVED_SYSTEM_PROMPT = r'''Eres asistente virtual del Vicerrectorado de Investigación UNA Puno.

IMPORTANTE: Si el usuario pide un formato de tesis (borrador o proyecto), 
NO respondas con texto. El sistema ya le enviará automáticamente el archivo PDF.

🎯 **INFORMACIÓN QUE MANEJAS:**
- Coordinadores por facultad (contactos exactos)
- Formatos de tesis en formato PDF (borrador/proyecto)
- Líneas y sublíneas de investigación  
- Manual de uso de la Plataforma para el tesista
- Procesos de tesis y reglamentos
- Preguntas frecuentes

🔍 **INSTRUCCIONES ESPECÍFICAS:**
1. Cuando el usuario pregunte por una facultad, proporciona TODA la información relevante del contexto
2. Si hay información de contacto da todo lo que tengas de esa facultad
3. sobre las líneas de investigación, incluye todas las sublíneas disponibles
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
            "• Formatos de tesis (borrador/proyecto)\n"
            "• Manual de uso de la plataforma para el tesista\n"
            "• Preguntas frecuentes sobre PGI\n\n"
            "💡 *Comandos útiles:*\n"
            "/ayuda - Ver esta información\n"
            "/reset - Reiniciar conversación\n\n"
            "⏱️ Sesión activa por 10 minutos\n"
            "🔒 Chat monitoreado para mejora continua\n\n"
            "¿En qué puedo ayudarte?"
        ), "welcome"
    
    if not context_docs:
        return "No encuentro información específica sobre ese tema en mi base de conocimiento actual. Te recomiendo contactar directamente con la coordinación de investigación de la facultad correspondiente.", "no_context"
    
    context_text = ""
    if context_docs:
        context_parts = []
        for doc in context_docs[:4]:
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
    
    full_prompt = IMPROVED_SYSTEM_PROMPT.format(
        context=context_text,
        user_query=user_message
    )
    
    response = await call_deepseek_async(full_prompt)
    
    if response:
        return response, DEEPSEEK_MODEL
    
    return "Disculpa, tengo dificultades técnicas en este momento. Por favor intenta nuevamente o contacta directamente al Vicerrectorado de Investigación.", "error"


# ============================================================================
# INACTIVITY CHECKER
# ============================================================================

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


# ============================================================================
# MESSAGE PROCESSING ⭐ ACTUALIZADO CON FORMATOS
# ============================================================================

async def process_message_async(user_message, phone_number):
    """Procesar mensaje con búsqueda optimizada y envío de formatos"""
    async with semaphore:
        start_time = time.time()
        
        # ⭐ PRIORIDAD 1: Verificar si pide un formato
        if FORMATOS_ENABLED:
            try:
                es_formato, enviado = await buscar_y_enviar_formato(user_message, phone_number)
                
                if es_formato:
                    if enviado:
                        # ✅ Formato enviado exitosamente
                        logger.info(f"📄 Formato enviado a {phone_number}")
                        return ""  # No enviar respuesta adicional
                    else:
                        # ❌ Era búsqueda de formato pero falló
                        # Ya se envió mensaje de error dentro de buscar_y_enviar_formato
                        logger.info(f"⚠️ Búsqueda de formato sin éxito para {phone_number}")
                        return ""  # No continuar con búsqueda normal
                
                # Si no es formato (False, False), continuar con flujo normal
                
            except Exception as e:
                logger.error(f"Error en búsqueda de formatos: {e}")
                # Si hay error, continuar con flujo normal
        
        # PRIORIDAD 2: Procesamiento normal del mensaje
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

        # Búsqueda optimizada con fallback
        loop = asyncio.get_event_loop()
        relevant_docs = await loop.run_in_executor(
            None, optimized_search_knowledge_base, user_message, 5, 0.3
        )
    
        # Fallback: Si no hay resultados, buscar directamente por facultad
        if not relevant_docs and any(word in user_message.lower() for word in 
                                ['línea', 'linea', 'investigación', 'investigacion', 'sublinea']):
            logger.info("   🔄 Usando búsqueda directa por facultad...")
            relevant_docs = await loop.run_in_executor(
                None, direct_faculty_search, user_message, documents, 3
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


# ============================================================================
# WHATSAPP HANDLER
# ============================================================================

def handle_incoming_message_sync(message):
    """Handler SYNC que WhatsAppAPIClient.start_polling() llama"""
    try:
        logger.info(f"📥 Mensaje recibido: {message}")
        
        if not isinstance(message, dict):
            logger.error(f"❌ Mensaje no es un diccionario: {type(message)}")
            return
        
        from_field = message.get('from', '')
        if not from_field:
            logger.error(f"❌ Campo 'from' vacío en mensaje: {message}")
            return
        
        phone_number = extract_phone_number(from_field)
        if not phone_number:
            logger.error(f"❌ No se pudo extraer número de: {from_field}")
            return
        
        user_message = message.get('body', '').strip()
        if not user_message:
            logger.warning(f"⚠️ Mensaje vacío de {phone_number}")
            return

        message_id = message.get('id')  # ✅ Extraer el ID
        
        logger.info(f"📨 {phone_number}: {user_message[:50]}")

        # Ejecutar tarea async en el event loop
        future = asyncio.run_coroutine_threadsafe(
            process_and_send(phone_number, user_message, message_id),  # ✅ Pasar el ID
            event_loop
        )
        
        time.sleep(RATE_LIMIT_DELAY)
        
    except Exception as e:
        logger.error(f"❌ Error handler: {e}", exc_info=True)

async def send_text_async(self, to: str, message: str) -> bool:
    """
    Enviar mensaje de texto (asíncrono)
    
    Args:
        to: Número de teléfono
        message: Mensaje a enviar
        
    Returns:
        True si se envió correctamente
    """
    try:
        url = f"{self.api_url}/api/whatsapp/send/text"
        payload = {
            'to': extract_phone_number(to),
            'message': message
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    logger.info(f"✅ Mensaje enviado a {to}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"❌ Error enviando mensaje: {response.status} - {text}")
                    return False
                
    except Exception as e:
        logger.error(f"❌ Excepción al enviar mensaje async: {str(e)}")
        return False

async def process_and_send(phone_number, user_message, message_id=None):
    """Procesar y enviar respuesta ⭐ ACTUALIZADO"""
    try:
        bot_response = await process_message_async(user_message, phone_number)
        
        # ⭐ NUEVO: Si la respuesta está vacía, ya se envió un formato
        if not bot_response or bot_response.strip() == "":
            logger.info(f"✅ Formato enviado directamente a {phone_number}")
            # Marcar como leído aunque no se envíe texto
            if message_id:
                await whatsapp_client.mark_message_as_read(message_id)
            return
        
        # Enviar respuesta normal (DEBE SER ASYNC)
        success = await whatsapp_client.send_text_async(phone_number, bot_response)
        
        # Marcar como leído después de enviar
        if message_id:
            await whatsapp_client.mark_message_as_read(message_id)
        
        if success:
            logger.info(f"✅ Enviado a {phone_number}")
        else:
            logger.error(f"❌ Error enviando a {phone_number}")
            
    except Exception as e:
        logger.error(f"❌ Error en process_and_send: {e}", exc_info=True)

# ============================================================================
# MAIN
# ============================================================================

async def main():
    global http_session, whatsapp_client, event_loop
    
    logger.info("=" * 60)
    logger.info("CHATBOT UNA PUNO - VERSIÓN CON ENVÍO DE PDFs")
    logger.info("✅ Búsqueda optimizada por facultades")
    logger.info("✅ Envío automático de formatos de tesis")
    logger.info("✅ Respuestas específicas con datos concretos")
    if FORMATOS_ENABLED:
        logger.info(f"✅ File Server: {FILE_SERVER_URL}")
    else:
        logger.info("⚠️  Envío de formatos DESACTIVADO")
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
    logger.info(f"🔍 Umbral similitud: 0.3")
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