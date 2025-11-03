"""
Servidor de archivos temporales para formatos de tesis
Sirve PDFs desde PostgreSQL con URLs temporales de 5 minutos
"""

import os
import asyncio
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import asyncpg
from quart import Quart, send_file, jsonify, abort
from quart_cors import cors
import io

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Quart(__name__)
app = cors(app, allow_origin="*")

# Configuración
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
PUBLIC_URL = os.getenv('PUBLIC_URL', 'https://files.vridevops.space')

# Cache temporal de URLs (token -> formato_id, expira_en)
temp_urls: Dict[str, tuple] = {}

# Pool de conexiones
db_pool = None


# ============================================================================
# INICIALIZACIÓN
# ============================================================================

async def init_db():
    """Inicializar pool de PostgreSQL"""
    global db_pool
    
    try:
        db_pool = await asyncpg.create_pool(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            min_size=2,
            max_size=10,
            command_timeout=30
        )
        logger.info("✅ PostgreSQL pool inicializado")
        return True
    except Exception as e:
        logger.error(f"❌ Error conectando a PostgreSQL: {e}")
        return False


def cleanup_expired_urls():
    """Limpiar URLs expiradas"""
    now = datetime.now()
    expired = [token for token, (_, expira) in temp_urls.items() if expira < now]
    
    for token in expired:
        del temp_urls[token]
    
    if expired:
        logger.info(f"🧹 Limpiados {len(expired)} tokens expirados")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
async def health():
    """Healthcheck"""
    db_status = 'ok' if db_pool else 'disconnected'
    
    return jsonify({
        'status': 'ok',
        'database': db_status,
        'urls_activas': len(temp_urls),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/temp-url/<int:formato_id>', methods=['POST'])
async def create_temp_url(formato_id: int):
    """
    Crear URL temporal para un formato
    
    POST /api/temp-url/123
    Response: {
        "url": "https://files.vridevops.space/download/abc123def",
        "filename": "formato.pdf",
        "expires_in": 300
    }
    """
    try:
        if not db_pool:
            return jsonify({'error': 'Base de datos no disponible'}), 503
        
        # Verificar que el formato existe
        result = await db_pool.fetchrow(
            "SELECT id, filename FROM formatos_tesis WHERE id = $1 AND activo = true",
            formato_id
        )
        
        if not result:
            logger.warning(f"Formato {formato_id} no encontrado")
            return jsonify({'error': 'Formato no encontrado'}), 404
        
        # Generar token único
        token = secrets.token_urlsafe(24)
        
        # Calcular expiración (5 minutos)
        expires_at = datetime.now() + timedelta(minutes=5)
        
        # Guardar en cache
        temp_urls[token] = (formato_id, expires_at)
        
        # Limpiar URLs expiradas
        cleanup_expired_urls()
        
        # Construir URL pública
        download_url = f"{PUBLIC_URL}/download/{token}"
        
        logger.info(f"✅ URL temporal creada para formato {formato_id}: {token[:8]}...")
        
        return jsonify({
            'url': download_url,
            'filename': result['filename'],
            'expires_in': 300,
            'expires_at': expires_at.isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error creando URL temporal: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@app.route('/download/<token>', methods=['GET'])
async def download_file(token: str):
    """
    Descargar archivo con token temporal
    
    GET /download/abc123def
    """
    try:
        # Verificar token
        if token not in temp_urls:
            logger.warning(f"Token inválido o expirado: {token[:8]}...")
            abort(404, "Link expirado o inválido")
        
        formato_id, expires_at = temp_urls[token]
        
        # Verificar expiración
        if datetime.now() > expires_at:
            del temp_urls[token]
            logger.warning(f"Token expirado: {token[:8]}...")
            abort(410, "Link expirado")
        
        if not db_pool:
            abort(503, "Base de datos no disponible")
        
        # Obtener archivo de DB
        result = await db_pool.fetchrow(
            "SELECT filename, file_data, mime_type FROM formatos_tesis WHERE id = $1",
            formato_id
        )
        
        if not result:
            logger.error(f"Archivo no encontrado para formato {formato_id}")
            abort(404, "Archivo no encontrado")
        
        # Enviar archivo
        file_stream = io.BytesIO(result['file_data'])
        filename = result['filename']
        mime_type = result['mime_type'] or 'application/pdf'
        
        # Eliminar token después de usar (one-time use)
        del temp_urls[token]
        
        logger.info(f"✅ Archivo descargado: {filename} (token: {token[:8]}...)")
        
        return await send_file(
            file_stream,
            mimetype=mime_type,
            as_attachment=True,
            attachment_filename=filename
        )
    
    except Exception as e:
        logger.error(f"Error descargando archivo: {e}")
        abort(500, "Error interno del servidor")


@app.route('/api/stats', methods=['GET'])
async def get_stats():
    """Estadísticas del servidor"""
    try:
        if not db_pool:
            return jsonify({'error': 'Base de datos no disponible'}), 503
        
        # Contar formatos
        total_formatos = await db_pool.fetchval(
            "SELECT COUNT(*) FROM formatos_tesis WHERE activo = true"
        )
        
        # Contar envíos
        total_envios = await db_pool.fetchval(
            "SELECT COUNT(*) FROM formatos_envios"
        )
        
        # Formatos más descargados
        top_formatos = await db_pool.fetch("""
            SELECT 
                ft.codigo,
                ft.titulo,
                COUNT(fe.id) as descargas
            FROM formatos_tesis ft
            LEFT JOIN formatos_envios fe ON ft.id = fe.formato_id
            WHERE ft.activo = true
            GROUP BY ft.id, ft.codigo, ft.titulo
            ORDER BY descargas DESC
            LIMIT 5
        """)
        
        return jsonify({
            'total_formatos': total_formatos,
            'total_envios': total_envios,
            'urls_activas': len(temp_urls),
            'top_formatos': [dict(f) for f in top_formatos]
        })
    
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.before_serving
async def startup():
    """Ejecutar al iniciar el servidor"""
    logger.info("="*60)
    logger.info("🗂️  FILE SERVER - FORMATOS DE TESIS")
    logger.info("="*60)
    logger.info(f"Host: {POSTGRES_HOST}")
    logger.info(f"Database: {POSTGRES_DB}")
    logger.info(f"Public URL: {PUBLIC_URL}")
    logger.info("="*60)
    
    success = await init_db()
    if not success:
        logger.error("❌ No se pudo conectar a PostgreSQL")
    else:
        logger.info("✅ Servidor listo")


@app.after_serving
async def shutdown():
    """Ejecutar al cerrar el servidor"""
    logger.info("🛑 Cerrando servidor...")
    if db_pool:
        await db_pool.close()
        logger.info("✅ Pool de PostgreSQL cerrado")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    
    logger.info(f"Iniciando en puerto {port}...")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )