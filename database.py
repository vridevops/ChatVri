"""
database.py
Módulo para manejar conexiones y operaciones con PostgreSQL
"""

import os
import logging
from datetime import datetime, date
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Pool de conexiones
connection_pool = None

def init_db_pool():
    """Inicializar pool de conexiones a PostgreSQL"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min y max conexiones
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            database=os.getenv('POSTGRES_DB', 'postgres'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            connect_timeout=10
        )
        logger.info("✅ Pool de conexiones PostgreSQL inicializado")
        return True
    except Exception as e:
        logger.error(f"❌ Error inicializando pool PostgreSQL: {e}")
        return False

@contextmanager
def get_db_connection():
    """Context manager para obtener conexión del pool"""
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error en transacción DB: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

# ==========================================
# FUNCIONES PARA USUARIOS
# ==========================================

def create_or_get_user(phone_number):
    """Crear o obtener usuario por número de teléfono"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO users (phone_number)
                    VALUES (%s)
                    ON CONFLICT (phone_number) 
                    DO UPDATE SET last_seen = CURRENT_TIMESTAMP
                    RETURNING id, phone_number, total_messages
                """, (phone_number,))
                return dict(cur.fetchone())
    except Exception as e:
        logger.error(f"Error creando/obteniendo usuario: {e}")
        return None

# ==========================================
# FUNCIONES PARA CONVERSACIONES
# ==========================================

def save_conversation(phone_number, user_message, bot_response, 
                     model_used="unknown", response_time_ms=0, 
                     tokens_used=0, context_length=0):
    """Guardar conversación en la base de datos"""
    try:
        user = create_or_get_user(phone_number)
        if not user:
            return False
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations 
                    (user_id, phone_number, user_message, bot_response, 
                     model_used, response_time_ms, tokens_used, context_length)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (user['id'], phone_number, user_message, bot_response,
                      model_used, response_time_ms, tokens_used, context_length))
                
        logger.info(f"✅ Conversación guardada: {phone_number}")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando conversación: {e}")
        return False

def get_user_conversation_history(phone_number, limit=5):
    """Obtener historial de conversaciones de un usuario"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
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

# ==========================================
# FUNCIONES PARA ESTADÍSTICAS
# ==========================================

def update_daily_stats():
    """Actualizar estadísticas diarias"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO daily_stats (date, total_messages, unique_users, avg_response_time_ms, total_tokens)
                    SELECT 
                        CURRENT_DATE,
                        COUNT(*),
                        COUNT(DISTINCT phone_number),
                        AVG(response_time_ms)::INT,
                        SUM(tokens_used)::INT
                    FROM conversations
                    WHERE DATE(created_at) = CURRENT_DATE
                    ON CONFLICT (date) DO UPDATE SET
                        total_messages = EXCLUDED.total_messages,
                        unique_users = EXCLUDED.unique_users,
                        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                        total_tokens = EXCLUDED.total_tokens
                """)
        return True
    except Exception as e:
        logger.error(f"Error actualizando estadísticas diarias: {e}")
        return False

def get_dashboard_stats(days=7):
    """Obtener estadísticas para el dashboard"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Stats generales
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(DISTINCT phone_number) as total_users,
                        AVG(response_time_ms)::INT as avg_response_time,
                        SUM(CASE WHEN DATE(created_at) = CURRENT_DATE THEN 1 ELSE 0 END) as today_messages
                    FROM conversations
                """)
                general_stats = dict(cur.fetchone())
                
                # Mensajes por hora (últimas 24h)
                cur.execute("""
                    SELECT 
                        EXTRACT(HOUR FROM created_at)::INT as hour,
                        COUNT(*) as count
                    FROM conversations
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY hour
                    ORDER BY hour
                """)
                messages_by_hour = [dict(row) for row in cur.fetchall()]
                
                # Top usuarios
                cur.execute("""
                    SELECT 
                        phone_number,
                        COUNT(*) as message_count
                    FROM conversations
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                    GROUP BY phone_number
                    ORDER BY message_count DESC
                    LIMIT 10
                """)
                top_users = [dict(row) for row in cur.fetchall()]
                
                # Conversaciones recientes
                cur.execute("""
                    SELECT 
                        phone_number,
                        user_message,
                        bot_response,
                        model_used,
                        response_time_ms,
                        created_at
                    FROM conversations
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                recent_conversations = [dict(row) for row in cur.fetchall()]
                
                return {
                    'general': general_stats,
                    'messages_by_hour': messages_by_hour,
                    'top_users': top_users,
                    'recent_conversations': recent_conversations
                }
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del dashboard: {e}")
        return None

# ==========================================
# FUNCIONES PARA BÚSQUEDAS
# ==========================================

def log_knowledge_search(phone_number, query, results_found, top_distance=None):
    """Registrar búsqueda en base de conocimiento"""
    try:
        user = create_or_get_user(phone_number)
        if not user:
            return False
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO knowledge_searches 
                    (user_id, query, results_found, top_distance)
                    VALUES (%s, %s, %s, %s)
                """, (user['id'], query, results_found, top_distance))
        return True
    except Exception as e:
        logger.error(f"Error registrando búsqueda: {e}")
        return False

# ==========================================
# FUNCIONES PARA LOGS DE ERRORES
# ==========================================

def log_error(error_type, error_message, phone_number=None, context=None):
    """Registrar errores en la base de datos"""
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

# ==========================================
# FUNCIONES PARA ADMIN
# ==========================================

def verify_admin_user(username, password):
    """Verificar credenciales de admin (simplificado)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, username, email
                    FROM admin_users
                    WHERE username = %s
                """, (username,))
                user = cur.fetchone()
                
                if user and password == 'unap2025':  # Simplificado para demo
                    # Actualizar last_login
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