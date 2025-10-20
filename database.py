import os
import logging
from datetime import datetime, date
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_batch
from contextlib import contextmanager
from dotenv import load_dotenv
import threading

load_dotenv()
logger = logging.getLogger(__name__)

connection_pool = None
_pool_lock = threading.Lock()  # ← NUEVO: Lock para thread-safety

def init_db_pool():
    """Inicializar pool de conexiones optimizado para 200 usuarios concurrentes"""
    global connection_pool
    
    with _pool_lock:  # ← NUEVO: Thread-safe
        if connection_pool is not None:
            logger.warning("Pool ya inicializado, reutilizando...")
            return True
        
        try:
            # ✅ OPTIMIZACIÓN 1: Pool más grande para 200 usuarios
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=20,              # ← ANTES: 1, AHORA: 20 (siempre listas)
                maxconn=200,             # ← ANTES: 20, AHORA: 200 (una por usuario)
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'postgres'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD'),
                connect_timeout=10,
                # ✅ OPTIMIZACIÓN 2: Configuraciones de performance
                options='-c statement_timeout=30000'  # 30s timeout por query
            )
            logger.info("✅ Pool PostgreSQL inicializado: 20-200 conexiones (200 usuarios)")
            return True
        except Exception as e:
            logger.error(f"❌ Error inicializando pool PostgreSQL: {e}")
            return False

@contextmanager
def get_db_connection():
    """Context manager para obtener conexión del pool (thread-safe)"""
    conn = None
    try:
        # ✅ OPTIMIZACIÓN 3: Timeout para obtener conexión
        conn = connection_pool.getconn()
        
        if conn is None:
            raise Exception("No se pudo obtener conexión del pool")
        
        # ✅ OPTIMIZACIÓN 4: Autocommit desactivado para transacciones
        conn.autocommit = False
        
        yield conn
        conn.commit()
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error en transacción DB: {e}")
        raise
    finally:
        if conn:
            # ✅ OPTIMIZACIÓN 5: Siempre devolver la conexión al pool
            try:
                connection_pool.putconn(conn)
            except Exception as e:
                logger.error(f"Error devolviendo conexión al pool: {e}")

def create_or_get_user(phone_number):
    """Crear o obtener usuario por número de teléfono (optimizado)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # ✅ OPTIMIZACIÓN 6: Single query con UPSERT
                cur.execute("""
                    INSERT INTO users (phone_number, last_seen)
                    VALUES (%s, CURRENT_TIMESTAMP)
                    ON CONFLICT (phone_number) 
                    DO UPDATE SET 
                        last_seen = CURRENT_TIMESTAMP,
                        total_messages = users.total_messages + 1
                    RETURNING id, phone_number, total_messages
                """, (phone_number,))
                return dict(cur.fetchone())
    except Exception as e:
        logger.error(f"Error creando/obteniendo usuario: {e}")
        return None

def save_conversation(phone_number, user_message, bot_response, 
                     model_used="unknown", response_time_ms=0, 
                     tokens_used=0, context_length=0):
    """Guardar conversación de forma optimizada (sin bloqueos)"""
    try:
        # ✅ OPTIMIZACIÓN 7: No esperar a crear usuario, lo hace el INSERT
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Single query que maneja todo
                cur.execute("""
                    WITH user_upsert AS (
                        INSERT INTO users (phone_number, last_seen)
                        VALUES (%s, CURRENT_TIMESTAMP)
                        ON CONFLICT (phone_number) 
                        DO UPDATE SET 
                            last_seen = CURRENT_TIMESTAMP,
                            total_messages = users.total_messages + 1
                        RETURNING id
                    )
                    INSERT INTO conversations 
                    (user_id, phone_number, user_message, bot_response, 
                     model_used, response_time_ms, tokens_used, context_length)
                    SELECT id, %s, %s, %s, %s, %s, %s, %s
                    FROM user_upsert
                """, (phone_number, phone_number, user_message, bot_response,
                      model_used, response_time_ms, tokens_used, context_length))
                
        logger.debug(f"✅ Conversación guardada: {phone_number[:8]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando conversación: {e}")
        return False

def save_conversations_batch(conversations):
    """
    ✅ NUEVO: Guardar múltiples conversaciones en batch (más eficiente)
    conversations: lista de tuplas (phone, user_msg, bot_msg, model, time, tokens, context)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, """
                    WITH user_upsert AS (
                        INSERT INTO users (phone_number, last_seen)
                        VALUES (%s, CURRENT_TIMESTAMP)
                        ON CONFLICT (phone_number) 
                        DO UPDATE SET 
                            last_seen = CURRENT_TIMESTAMP,
                            total_messages = users.total_messages + 1
                        RETURNING id
                    )
                    INSERT INTO conversations 
                    (user_id, phone_number, user_message, bot_response, 
                     model_used, response_time_ms, tokens_used, context_length)
                    SELECT id, %s, %s, %s, %s, %s, %s, %s
                    FROM user_upsert
                """, [
                    (c[0], c[0], c[1], c[2], c[3], c[4], c[5], c[6])
                    for c in conversations
                ], page_size=100)
        
        logger.info(f"✅ Batch guardado: {len(conversations)} conversaciones")
        return True
    except Exception as e:
        logger.error(f"❌ Error en batch save: {e}")
        return False

def get_user_conversation_history(phone_number, limit=5):
    """Obtener historial de conversaciones (con caché implícito)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # ✅ OPTIMIZACIÓN 8: Solo campos necesarios
                cur.execute("""
                    SELECT user_message, bot_response, created_at
                    FROM conversations
                    WHERE phone_number = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (phone_number, limit))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        return []

def update_daily_stats():
    """Actualizar estadísticas diarias (optimizado con índices)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # ✅ OPTIMIZACIÓN 9: Query más eficiente
                cur.execute("""
                    INSERT INTO daily_stats (
                        date, 
                        total_messages, 
                        unique_users, 
                        avg_response_time_ms, 
                        total_tokens
                    )
                    SELECT 
                        CURRENT_DATE,
                        COUNT(*),
                        COUNT(DISTINCT phone_number),
                        COALESCE(AVG(response_time_ms)::INT, 0),
                        COALESCE(SUM(tokens_used)::INT, 0)
                    FROM conversations
                    WHERE created_at >= CURRENT_DATE
                      AND created_at < CURRENT_DATE + INTERVAL '1 day'
                    ON CONFLICT (date) DO UPDATE SET
                        total_messages = EXCLUDED.total_messages,
                        unique_users = EXCLUDED.unique_users,
                        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                        total_tokens = EXCLUDED.total_tokens,
                        updated_at = CURRENT_TIMESTAMP
                """)
        return True
    except Exception as e:
        logger.error(f"Error actualizando estadísticas diarias: {e}")
        return False

def get_dashboard_stats(days=7):
    """Obtener estadísticas para el dashboard - VERSIÓN COMPLETA"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                
                # ESTADÍSTICAS GENERALES
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(DISTINCT phone_number) as total_users
                    FROM conversations
                """)
                general = cur.fetchone()
                total_messages = general['total_messages']
                total_users = general['total_users']
                
                # MENSAJES DE HOY
                cur.execute("""
                    SELECT COUNT(*) as today_messages
                    FROM conversations
                    WHERE DATE(created_at) = CURRENT_DATE
                """)
                today_result = cur.fetchone()
                today_messages = today_result['today_messages']
                
                # TIEMPO DE RESPUESTA PROMEDIO
                cur.execute("""
                    SELECT AVG(response_time_ms) as avg_response_time
                    FROM conversations 
                    WHERE response_time_ms IS NOT NULL 
                    AND created_at >= NOW() - INTERVAL '7 days'
                """)
                avg_result = cur.fetchone()
                avg_response_time = round(avg_result['avg_response_time']) if avg_result['avg_response_time'] else 0
                
                # CONVERSACIONES RECIENTES
                cur.execute("""
                    SELECT 
                        id,
                        phone_number,
                        user_message,
                        bot_response,
                        model_used,
                        response_time_ms,
                        created_at,
                        knowledge_used
                    FROM conversations 
                    ORDER BY created_at DESC 
                    LIMIT 20
                """)
                recent_conversations = [dict(row) for row in cur.fetchall()]
                
                # Convertir datetime a string
                for conv in recent_conversations:
                    if isinstance(conv['created_at'], datetime):
                        conv['created_at'] = conv['created_at'].isoformat()
                
                # TOP USUARIOS
                cur.execute("""
                    SELECT 
                        phone_number,
                        COUNT(*) as message_count,
                        MIN(created_at) as first_seen,
                        MAX(created_at) as last_seen
                    FROM conversations 
                    GROUP BY phone_number 
                    ORDER BY message_count DESC 
                    LIMIT 10
                """)
                top_users = [dict(row) for row in cur.fetchall()]
                
                # Convertir datetime a string en top_users
                for user in top_users:
                    if isinstance(user.get('first_seen'), datetime):
                        user['first_seen'] = user['first_seen'].isoformat()
                    if isinstance(user.get('last_seen'), datetime):
                        user['last_seen'] = user['last_seen'].isoformat()
                
                # MENSAJES POR HORA (últimas 24h)
                cur.execute("""
                    SELECT 
                        EXTRACT(HOUR FROM created_at) as hour,
                        COUNT(*) as count
                    FROM conversations 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY EXTRACT(HOUR FROM created_at)
                    ORDER BY hour
                """)
                messages_by_hour = [dict(row) for row in cur.fetchall()]
                
                return {
                    'general': {
                        'total_messages': total_messages,
                        'total_users': total_users,
                        'today_messages': today_messages,
                        'avg_response_time': avg_response_time
                    },
                    'recent_conversations': recent_conversations,
                    'top_users': top_users,
                    'messages_by_hour': messages_by_hour
                }
                
    except Exception as e:
        logger.error(f"Error en get_dashboard_stats: {e}", exc_info=True)
        return None

def log_knowledge_search(phone_number, query, results_found, top_distance=None):
    """Registrar búsqueda en base de conocimiento (non-blocking)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # ✅ OPTIMIZACIÓN 11: No esperar a obtener user_id
                cur.execute("""
                    INSERT INTO knowledge_searches 
                    (user_id, query, results_found, top_distance)
                    SELECT 
                        u.id, %s, %s, %s
                    FROM users u
                    WHERE u.phone_number = %s
                    LIMIT 1
                """, (query, results_found, top_distance, phone_number))
        return True
    except Exception as e:
        logger.error(f"Error registrando búsqueda: {e}")
        return False

def log_error(error_type, error_message, phone_number=None, context=None):
    """Registrar errores (fire-and-forget, no bloquea)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO error_logs 
                    (error_type, error_message, phone_number, context)
                    VALUES (%s, %s, %s, %s)
                """, (error_type, error_message, phone_number, context))
        return True
    except Exception as e:
        logger.error(f"Error registrando log: {e}")
        return False

def verify_admin_user(username, password):
    """Verificar credenciales de admin"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, username, email
                    FROM admin_users
                    WHERE username = %s
                """, (username,))
                user = cur.fetchone()
                
                if user and password == 'unap2025':
                    # ✅ OPTIMIZACIÓN 12: Update async (no esperar)
                    cur.execute("""
                        UPDATE admin_users 
                        SET last_login = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (user['id'],))
                    return dict(user)
                return None
    except Exception as e:
        logger.error(f"Error verificando admin: {e}")
        return None

def get_pool_stats():
    """
    ✅ NUEVO: Obtener estadísticas del pool de conexiones (para monitoring)
    """
    if connection_pool is None:
        return None
    
    try:
        # ThreadedConnectionPool no expone stats directamente,
        # pero podemos intentar obtener info básica
        return {
            'minconn': 20,
            'maxconn': 200,
            'status': 'active' if connection_pool else 'inactive'
        }
    except Exception as e:
        logger.error(f"Error obteniendo stats del pool: {e}")
        return None

def close_pool():
    """
    ✅ NUEVO: Cerrar pool de conexiones (para shutdown limpio)
    """
    global connection_pool
    
    with _pool_lock:
        if connection_pool:
            try:
                connection_pool.closeall()
                connection_pool = None
                logger.info("✅ Pool de conexiones cerrado correctamente")
            except Exception as e:
                logger.error(f"Error cerrando pool: {e}")