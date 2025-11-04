"""
Cliente Python para API de WhatsApp
Reemplaza Twilio en el chatbot
"""

import requests
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
        
    def _get_headers(self) -> dict:
        """Headers comunes para todas las peticiones"""
        return {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
    
    def send_text(self, to: str, message: str) -> bool:
        """
        Enviar mensaje de texto
        
        Args:
            to: Número de teléfono (formato: 51987654321)
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
            
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Mensaje enviado a {to}")
                return True
            else:
                logger.error(f"❌ Error enviando mensaje: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Excepción al enviar mensaje: {str(e)}")
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
    
    def start_polling(self, callback, interval: int = 3):
        """
        Iniciar polling de mensajes
        
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
                        
                        # Llamar al callback con el mensaje
                        callback(msg)
                
                # Limpiar mensajes procesados viejos (mantener solo últimos 1000)
                if len(self.processed_messages) > 1000:
                    self.processed_messages = set(list(self.processed_messages)[-1000:])
                
            except Exception as e:
                logger.error(f"❌ Error en polling: {str(e)}")
            
            time.sleep(interval)


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