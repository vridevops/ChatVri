"""
Cliente Python para API de WhatsApp
Versión completa con soporte async
"""

import requests
import aiohttp
import time
import logging
import re
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class WhatsAppAPIClient:
    """Cliente para interactuar con la API de WhatsApp"""
    
    def __init__(self, api_url: str, api_key: str):
        """
        Inicializar cliente
        
        Args:
            api_url: URL base de la API (ej: https://apiwsp.services.vridevops.space)
            api_key: API Key configurada en .env
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.processed_messages = set()
        self.last_check = None
        self.session = None  # Para requests async
        
    def _get_headers(self) -> dict:
        """Headers comunes para todas las peticiones"""
        return {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
    
    def check_connection(self) -> bool:
        """
        Verificar conexión con la API de WhatsApp
        """
        try:
            # Probar diferentes endpoints posibles
            endpoints_to_try = [
                "/api/whatsapp/status",
                "/api/status",
                "/status",
                "/health"
            ]
            
            for endpoint in endpoints_to_try:
                url = f"{self.api_url}{endpoint}"
                logger.info(f"🔍 Probando: {url}")
                
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=10
                )
                
                logger.info(f"📡 Status Code: {response.status_code}")
                logger.info(f"📄 Response: {response.text[:200]}")  # Primeros 200 caracteres
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ JSON Response: {data}")
                    
                    # Probar diferentes estructuras de respuesta
                    is_connected = (
                        data.get('connected') or 
                        data.get('status') == 'connected' or
                        data.get('ready') or
                        'connected' in str(data).lower()
                    )
                    
                    if is_connected:
                        logger.info("✅ WhatsApp conectado correctamente")
                        return True
            
            logger.warning("⚠️ Ningún endpoint respondió correctamente")
            return False
                    
        except Exception as e:
            logger.error(f"❌ Excepción al verificar conexión: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
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
    
    async def send_media_async(self, to: str, media_url: str, caption: str = "") -> bool:
        """
        Enviar archivo multimedia (PDF, imagen, etc.)
        
        Args:
            to: Número de teléfono
            media_url: URL del archivo a enviar
            caption: Texto opcional
            
        Returns:
            True si se envió correctamente
        """
        try:
            url = f"{self.api_url}/api/whatsapp/send/media"
            payload = {
                'to': extract_phone_number(to),
                'mediaUrl': media_url,
                'caption': caption
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Media enviado a {to}")
                        return True
                    else:
                        text = await response.text()
                        logger.error(f"❌ Error enviando media: {response.status} - {text}")
                        return False
                    
        except Exception as e:
            logger.error(f"❌ Excepción al enviar media: {str(e)}")
            return False
    
    def get_messages(self, limit: int = 50) -> List[Dict]:
        """
        Obtener mensajes recientes no leídos
        
        Args:
            limit: Cantidad máxima de mensajes a obtener
            
        Returns:
            Lista de mensajes
        """
        try:
            url = f"{self.api_url}/api/whatsapp/messages"
            params = {'limit': limit, 'unreadOnly': 'true'}
            
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                messages = response.json().get('messages', [])
                return messages
            else:
                logger.error(f"❌ Error obteniendo mensajes: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Excepción al obtener mensajes: {str(e)}")
            return []
    
    async def get_messages_async(self, limit: int = 50) -> List[Dict]:
        """
        Obtener mensajes recientes no leídos (asíncrono)
        
        Args:
            limit: Cantidad máxima de mensajes a obtener
            
        Returns:
            Lista de mensajes
        """
        try:
            url = f"{self.api_url}/api/whatsapp/messages"
            params = {'limit': limit, 'unreadOnly': 'true'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('messages', [])
                    else:
                        logger.error(f"❌ Error obteniendo mensajes: {response.status}")
                        return []
                    
        except Exception as e:
            logger.error(f"❌ Excepción al obtener mensajes async: {str(e)}")
            return []
    
    def start_polling(self, callback, interval: int = 3):
        """
        Iniciar polling de mensajes (síncrono)
        
        Args:
            callback: Función a llamar cuando llegue un mensaje
            interval: Intervalo en segundos entre cada revisión
        """
        logger.info(f"🔄 Iniciando polling cada {interval} segundos...")
        
        while True:
            try:
                messages = self.get_messages()
                
                for msg in messages:
                    msg_id = msg.get('id')
                    
                    # Evitar procesar mensajes duplicados
                    if msg_id and msg_id not in self.processed_messages:
                        self.processed_messages.add(msg_id)
                        callback(msg)
                
                # Limpiar mensajes procesados viejos
                if len(self.processed_messages) > 1000:
                    self.processed_messages = set(list(self.processed_messages)[-1000:])
                
            except Exception as e:
                logger.error(f"❌ Error en polling: {str(e)}")
            
            time.sleep(interval)

async def send_media_url(self, phone: str, media_url: str, caption: str = "") -> bool:
    """
    Enviar archivo por URL (async)
    
    Args:
        phone: Número de teléfono
        media_url: URL pública del archivo
        caption: Texto que acompaña el archivo
    
    Returns:
        bool: True si se envió exitosamente
    """
    try:
        import aiohttp
        
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.api_url}/api/whatsapp/send/media-url"
        
        payload = {
            'phone': phone,
            'mediaUrl': media_url,
            'caption': caption
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                success = resp.status == 200
                if success:
                    logger.info(f"✅ Archivo enviado a {phone}")
                else:
                    error_text = await resp.text()
                    logger.error(f"❌ Error enviando archivo: {resp.status} - {error_text}")
                return success
    
    except Exception as e:
        logger.error(f"Error en send_media_url: {e}")
        return False


def extract_phone_number(phone: str) -> str:
    """
    Extrae y formatea un número de teléfono al formato internacional
    
    Args:
        phone: Número de teléfono en cualquier formato
    
    Returns:
        Número de teléfono formateado (solo dígitos)
    
    Example:
        >>> extract_phone_number("+51 987 654 321")
        '51987654321'
        >>> extract_phone_number("whatsapp:+51987654321")
        '51987654321'
    """
    # Eliminar todo excepto dígitos
    cleaned = re.sub(r'\D', '', phone)
    
    # Si no empieza con código de país, asumir Perú (+51)
    if not cleaned.startswith('51') and len(cleaned) == 9:
        cleaned = '51' + cleaned
    
    return cleaned

