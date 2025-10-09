"""
Dashboard web interactivo para monitorear el chatbot en tiempo real
Lee estadísticas desde archivo compartido
"""
from flask import Flask, render_template_string, jsonify
from datetime import datetime
import json
from pathlib import Path
from collections import Counter

app = Flask(__name__)

# HTML del Dashboard (mismo que antes)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - Chatbot UNA Puno</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            text-align: center;
            position: relative;
        }
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .last-update {
            position: absolute;
            top: 15px;
            right: 20px;
            color: #999;
            font-size: 0.85em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .stat-card h3 {
            color: #667eea;
            font-size: 1em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-card .value {
            color: #333;
            font-size: 2.5em;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stat-card .label {
            color: #999;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .stat-card.updated .value {
            animation: pulse 0.5s ease-in-out;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); color: #667eea; }
        }
        .section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .section h2 {
            color: #333;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .conversation {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .conversation .time {
            color: #999;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .conversation .user {
            color: #667eea;
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .conversation .message {
            color: #333;
            margin-bottom: 10px;
            padding: 8px;
            background: white;
            border-radius: 5px;
        }
        .conversation .response {
            color: #666;
            padding: 8px;
            padding-left: 20px;
            border-left: 2px solid #ddd;
            background: white;
            border-radius: 5px;
        }
        .query-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }
        .query-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        .query-text {
            flex: 1;
            color: #333;
        }
        .badge {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        }
        .status {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 2s infinite;
        }
        .status.online { background: #28a745; }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .refresh-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
            margin-left: 10px;
            animation: pulse-dot 1.5s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.3;
        }
        .progress-bar {
            height: 8px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="last-update">
                Última actualización: <span id="last-update">--:--:--</span>
                <span class="refresh-indicator"></span>
            </div>
            <h1>🤖 Dashboard Chatbot WhatsApp</h1>
            <p>Vicerrectorado de Investigación - UNA Puno</p>
            <p style="margin-top: 10px;">
                <span class="status online"></span>
                <strong>Sistema Activo</strong>
            </p>
        </div>

        <div id="error-container"></div>

        <div class="stats-grid">
            <div class="stat-card" id="card-users">
                <h3>👥 Usuarios Activos</h3>
                <div class="value" id="total-users">0</div>
                <div class="label">Total de conversaciones</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="users-progress" style="width: 0%"></div>
                </div>
            </div>
            <div class="stat-card" id="card-messages">
                <h3>💬 Mensajes</h3>
                <div class="value" id="total-messages">0</div>
                <div class="label">Total procesados</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="messages-progress" style="width: 0%"></div>
                </div>
            </div>
            <div class="stat-card" id="card-docs">
                <h3>📚 Base de Conocimiento</h3>
                <div class="value" id="kb-docs">0</div>
                <div class="label">Documentos indexados</div>
            </div>
            <div class="stat-card" id="card-model">
                <h3>🤖 Modelo IA</h3>
                <div class="value" id="model-status" style="font-size: 1.3em;">Cargando...</div>
                <div class="label">Modelo activo</div>
            </div>
        </div>

        <div class="section">
            <h2>
                <span>📊 Consultas Más Frecuentes</span>
                <span style="font-size: 0.6em; color: #999;">Top 10</span>
            </h2>
            <div id="top-queries">
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <p>Esperando consultas...</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>
                <span>💬 Últimas Conversaciones</span>
                <span style="font-size: 0.6em; color: #999;">Tiempo real</span>
            </h2>
            <div id="recent-conversations">
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
                    </svg>
                    <p>Sin conversaciones aún...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let previousStats = {
            users: 0,
            messages: 0
        };

        function updateStats() {
            fetch('/api/stats')
                .then(response => {
                    if (!response.ok) throw new Error('Error al obtener estadísticas');
                    return response.json();
                })
                .then(data => {
                    // Limpiar mensajes de error
                    document.getElementById('error-container').innerHTML = '';
                    
                    // Actualizar timestamp
                    const now = new Date();
                    document.getElementById('last-update').textContent = now.toLocaleTimeString('es-PE');
                    
                    // Actualizar estadísticas con animación
                    updateValue('total-users', data.total_users, 'card-users');
                    updateValue('total-messages', data.total_messages, 'card-messages');
                    updateValue('kb-docs', data.kb_documents, 'card-docs');
                    
                    // Actualizar barras de progreso
                    updateProgress('users-progress', data.total_users, 50);
                    updateProgress('messages-progress', data.total_messages, 100);
                    
                    // Actualizar modelo
                    document.getElementById('model-status').textContent = data.model_used;
                    
                    // Actualizar top queries
                    updateTopQueries(data.top_queries);
                    
                    // Actualizar conversaciones
                    updateConversations(data.recent_conversations);
                    
                    // Guardar valores anteriores
                    previousStats.users = data.total_users;
                    previousStats.messages = data.total_messages;
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('error-container').innerHTML = `
                        <div class="error-message">
                            ⚠️ <strong>Error:</strong> No se pueden cargar las estadísticas. 
                            Asegúrate de que el chatbot esté ejecutándose (python main.py)
                        </div>
                    `;
                });
        }

        function updateValue(elementId, newValue, cardId) {
            const element = document.getElementById(elementId);
            const oldValue = parseInt(element.textContent) || 0;
            
            if (newValue !== oldValue) {
                const card = document.getElementById(cardId);
                card.classList.add('updated');
                animateValue(element, oldValue, newValue, 500);
                setTimeout(() => card.classList.remove('updated'), 500);
            }
        }

        function animateValue(element, start, end, duration) {
            const range = end - start;
            const increment = range / (duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                    current = end;
                    clearInterval(timer);
                }
                element.textContent = Math.floor(current);
            }, 16);
        }

        function updateProgress(elementId, value, max) {
            const percentage = Math.min((value / max) * 100, 100);
            document.getElementById(elementId).style.width = percentage + '%';
        }

        function updateTopQueries(queries) {
            const container = document.getElementById('top-queries');
            
            if (queries.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        <p>Esperando consultas...</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            queries.forEach(([query, count]) => {
                html += `
                    <div class="query-item">
                        <span class="query-text">${escapeHtml(query)}</span>
                        <span class="badge">${count}</span>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function updateConversations(conversations) {
            const container = document.getElementById('recent-conversations');
            
            if (conversations.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
                    </svg>
                        <p>Sin conversaciones aún...</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            conversations.forEach(conv => {
                const shortResponse = conv.bot_msg.length > 150 
                    ? conv.bot_msg.substring(0, 150) + '...' 
                    : conv.bot_msg;
                    
                html += `
                    <div class="conversation">
                        <div class="time">📅 ${conv.time}</div>
                        <div class="user">👤 ${conv.phone}</div>
                        <div class="message"><strong>Usuario:</strong> ${escapeHtml(conv.user_msg)}</div>
                        <div class="response"><strong>Bot:</strong> ${escapeHtml(shortResponse)}</div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Actualizar cada 3 segundos
        updateStats();
        setInterval(updateStats, 3000);
    </script>
</body>
</html>
"""

def load_stats_from_file():
    """Cargar estadísticas desde archivo compartido"""
    try:
        stats_file = Path('dashboard_stats.json')
        
        if not stats_file.exists():
            # Retornar stats vacías si no existe el archivo
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'total_users': 0,
                'total_messages': 0,
                'kb_documents': 0,
                'model_used': 'N/A',
                'top_queries': [],
                'recent_conversations': []
            }
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Procesar datos
        stats = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'total_users': data.get('total_users', 0),
            'total_messages': data.get('total_messages', 0),
            'kb_documents': data.get('kb_documents', 0),
            'model_used': data.get('model_used', 'N/A'),
            'top_queries': [],
            'recent_conversations': []
        }
        
        # Procesar top queries
        all_queries = []
        for conv in data.get('conversations', []):
            query = conv.get('user', '').lower().strip()
            if len(query) > 3:
                all_queries.append(query)
        
        if all_queries:
            query_counts = Counter(all_queries)
            stats['top_queries'] = query_counts.most_common(10)
        
        # Procesar conversaciones recientes
        conversations = data.get('conversations', [])
        for conv in sorted(conversations, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]:
            stats['recent_conversations'].append({
                'phone': '***' + conv.get('phone', '')[-7:],
                'time': conv.get('timestamp', '')[:19].replace('T', ' '),
                'user_msg': conv.get('user', ''),
                'bot_msg': conv.get('bot', '')
            })
        
        return stats
        
    except Exception as e:
        print(f"Error cargando estadísticas: {e}")
        return {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'total_users': 0,
            'total_messages': 0,
            'kb_documents': 0,
            'model_used': 'Error',
            'top_queries': [],
            'recent_conversations': []
        }

@app.route('/')
def dashboard():
    """Página principal del dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/stats')
def api_stats():
    """API para obtener estadísticas en JSON"""
    return jsonify(load_stats_from_file())

@app.route('/api/health')
def health():
    """Endpoint de salud del dashboard"""
    stats_file = Path('dashboard_stats.json')
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'dashboard_active': True,
        'stats_file_exists': stats_file.exists()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🎯 DASHBOARD DEL CHATBOT - ACTUALIZACIÓN AUTOMÁTICA")
    print("=" * 70)
    print(f"📊 Dashboard principal: http://localhost:5001")
    print(f"🔌 API estadísticas:    http://localhost:5001/api/stats")
    print(f"💚 Health check:        http://localhost:5001/api/health")
    print("=" * 70)
    print("✨ El dashboard se actualiza automáticamente cada 3 segundos")
    print("⚠️  IMPORTANTE: Asegúrate de que main.py esté corriendo")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5001, debug=False)