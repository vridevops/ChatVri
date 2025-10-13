"""
api_dashboard.py
API REST para el dashboard del chatbot
Ejecutar con: python api_dashboard.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from database import (
    init_db_pool,
    get_dashboard_stats,
    verify_admin_user
)

load_dotenv()

app = Flask(__name__)
CORS(app)  # Permitir CORS para el frontend

# Inicializar DB
init_db_pool()

@app.route('/api/health', methods=['GET'])
def health_check():
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
    """Obtener conversaciones con filtros"""
    from database import get_db_connection
    from psycopg2.extras import RealDictCursor
    
    phone = request.args.get('phone')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT 
                        phone_number,
                        user_message,
                        bot_response,
                        model_used,
                        response_time_ms,
                        created_at
                    FROM conversations
                """
                params = []
                
                if phone:
                    query += " WHERE phone_number = %s"
                    params.append(phone)
                
                query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(query, params)
                conversations = [dict(row) for row in cur.fetchall()]
                
                # Convertir datetime a string para JSON
                for conv in conversations:
                    conv['created_at'] = conv['created_at'].isoformat()
                
                return jsonify(conversations), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Obtener lista de usuarios"""
    from database import get_db_connection
    from psycopg2.extras import RealDictCursor
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        phone_number,
                        total_messages,
                        first_seen,
                        last_seen,
                        is_active
                    FROM users
                    ORDER BY total_messages DESC
                    LIMIT 100
                """)
                users = [dict(row) for row in cur.fetchall()]
                
                # Convertir datetime a string
                for user in users:
                    user['first_seen'] = user['first_seen'].isoformat()
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