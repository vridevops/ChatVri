from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime
from dotenv import load_dotenv
from database import (
    init_db_pool,
    get_dashboard_stats,
    verify_admin_user,
    get_db_connection
)
from psycopg2.extras import RealDictCursor

load_dotenv()

app = Flask(__name__)
CORS(app)

# Inicializar DB
init_db_pool()

# ✅ ENDPOINT HEALTH - AGREGADO PARA HEALTHCHECK
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para healthcheck de Docker/Coolify"""
    return jsonify({'status': 'healthy', 'service': 'chatbot-dashboard-api'}), 200

@app.route('/api/health', methods=['GET'])
def api_health_check():
    """Verificar que la API esté funcionando"""
    return jsonify({'status': 'ok', 'service': 'chatbot-dashboard-api'}), 200

@app.route('/api/login', methods=['POST'])
def login():
    """Endpoint de login"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username y password requeridos'}), 400
    
    user = verify_admin_user(username, password)
    
    if user:
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user.get('email')
            }
        }), 200
    else:
        return jsonify({'error': 'Credenciales incorrectas'}), 401

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtener estadísticas del dashboard"""
    days = request.args.get('days', 7, type=int)
    
    stats = get_dashboard_stats(days)
    
    if stats:
        return jsonify(stats), 200
    else:
        return jsonify({'error': 'Error obteniendo estadísticas'}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Obtener conversaciones con filtros - ENDPOINT MEJORADO"""
    phone = request.args.get('phone')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if phone:
                    # Filtrar por número específico
                    query = """
                        SELECT 
                            phone_number,
                            user_message,
                            bot_response,
                            model_used,
                            response_time_ms,
                            created_at
                        FROM conversations
                        WHERE phone_number = %s
                        ORDER BY created_at ASC
                        LIMIT %s OFFSET %s
                    """
                    cur.execute(query, (phone, limit, offset))
                else:
                    # Todas las conversaciones (más recientes primero)
                    query = """
                        SELECT 
                            phone_number,
                            user_message,
                            bot_response,
                            model_used,
                            response_time_ms,
                            created_at
                        FROM conversations
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    cur.execute(query, (limit, offset))
                
                conversations = [dict(row) for row in cur.fetchall()]
                
                # Convertir datetime a string
                for conv in conversations:
                    if isinstance(conv['created_at'], datetime):
                        conv['created_at'] = conv['created_at'].isoformat()
                
                return jsonify(conversations), 200
                
    except Exception as e:
        app.logger.error(f"Error obteniendo conversaciones: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/count', methods=['GET'])
def get_conversations_count():
    """Obtener conteo de conversaciones por usuario"""
    phone = request.args.get('phone')
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if phone:
                    cur.execute("""
                        SELECT COUNT(*) as total
                        FROM conversations
                        WHERE phone_number = %s
                    """, (phone,))
                else:
                    cur.execute("""
                        SELECT COUNT(*) as total
                        FROM conversations
                    """)
                
                result = cur.fetchone()
                return jsonify({'total': result['total']}), 200
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Obtener lista de usuarios únicos con sus estadísticas"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        u.phone_number,
                        u.total_messages,
                        u.first_seen,
                        u.last_seen,
                        u.is_active,
                        COUNT(c.id) as conversation_count
                    FROM users u
                    LEFT JOIN conversations c ON u.phone_number = c.phone_number
                    GROUP BY u.id, u.phone_number, u.total_messages, u.first_seen, u.last_seen, u.is_active
                    ORDER BY u.last_seen DESC
                    LIMIT 100
                """)
                users = [dict(row) for row in cur.fetchall()]
                
                # Convertir datetime a string
                for user in users:
                    if isinstance(user.get('first_seen'), datetime):
                        user['first_seen'] = user['first_seen'].isoformat()
                    if isinstance(user.get('last_seen'), datetime):
                        user['last_seen'] = user['last_seen'].isoformat()
                
                return jsonify(users), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('DASHBOARD_API_PORT', '5000'))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('FLASK_ENV') == 'development'
    )