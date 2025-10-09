"""
Cliente Python para API de WhatsApp
Reemplaza Twilio en el chatbot
"""

import requests
import time
import logging
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class WhatsAppAPIClient:
    """Cliente para interactuar con la API de WhatsApp"""
    
    def __init__(self, api_url: str, api_key: str):
        """
        Inicializar cliente
        
        Args:
            api_url: URL base de la API (ej: http://localhost:3000)
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
                'to': self._format_phone(to),
                'message': message
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Mensaje enviado a {to}")
                return True
            else:
                logger.error(f"Error enviando mensaje: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error en send_text: {e}")
            return False
    
    def send_image(self, to: str, image_url: str, caption: str = '') -> bool:
        """
        Enviar imagen desde URL
        
        Args:
            to: Número de teléfono
            image_url: URL de la imagen
            caption: Texto adicional (opcional)
            
        Returns:
            True si se envió correctamente
        """
        try:
            url = f"{self.api_url}/api/whatsapp/send/media-url"
            payload = {
                'to': self._format_phone(to),
                'media': image_url,
                'caption': caption
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error en send_image: {e}")
            return False
    
    def send_location(self, to: str, latitude: float, longitude: float, 
                     description: str = '') -> bool:
        """
        Enviar ubicación
        
        Args:
            to: Número de teléfono
            latitude: Latitud
            longitude: Longitud
            description: Descripción del lugar
            
        Returns:
            True si se envió correctamente
        """
        try:
            url = f"{self.api_url}/api/whatsapp/send/location"
            payload = {
                'to': self._format_phone(to),
                'latitude': latitude,
                'longitude': longitude,
                'description': description
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error en send_location: {e}")
            return False
    
    def get_new_messages(self, limit: int = 50) -> List[Dict]:
        """
        Obtener mensajes nuevos desde la API
        
        Args:
            limit: Número máximo de mensajes a obtener
            
        Returns:
            Lista de mensajes nuevos
        """
        try:
            url = f"{self.api_url}/api/whatsapp/messages"
            params = {'limit': limit}
            
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Error obteniendo mensajes: {response.status_code}")
                return []
            
            data = response.json()
            all_messages = data.get('data', [])
            
            # Filtrar solo mensajes nuevos entrantes (no enviados por nosotros)
            new_messages = []
            for msg in all_messages:
                msg_id = msg.get('id')
                from_user = msg.get('from', '')
                
                # Solo mensajes de usuarios (terminan en @c.us)
                if '@c.us' in from_user and msg_id not in self.processed_messages:
                    self.processed_messages.add(msg_id)
                    new_messages.append(msg)
            
            # Limpiar mensajes procesados antiguos (mantener últimos 1000)
            if len(self.processed_messages) > 1000:
                self.processed_messages = set(list(self.processed_messages)[-1000:])
            
            return new_messages
            
        except Exception as e:
            logger.error(f"Error en get_new_messages: {e}")
            return []
    
    def check_connection(self) -> bool:
        """
        Verificar si WhatsApp está conectado
        
        Returns:
            True si está conectado
        """
        try:
            url = f"{self.api_url}/api/whatsapp/status"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get('connected', False)
            
            return False
            
        except Exception as e:
            logger.error(f"Error verificando conexión: {e}")
            return False
    
    def _format_phone(self, phone: str) -> str:
        """
        Formatear número de teléfono
        Remueve @c.us, +, y otros caracteres
        
        Args:
            phone: Número de teléfono en cualquier formato
            
        Returns:
            Número limpio (ej: 51987654321)
        """
        # Remover @c.us si existe
        if '@c.us' in phone:
            phone = phone.replace('@c.us', '')
        
        # Remover espacios y caracteres especiales
        phone = ''.join(filter(str.isdigit, phone))
        
        return phone
    
    def start_polling(self, callback, interval: int = 2):
        """
        Iniciar polling continuo para recibir mensajes
        
        Args:
            callback: Función a llamar con cada mensaje nuevo
                     Debe aceptar un dict con el mensaje
            interval: Segundos entre cada verificación
        """
        logger.info(f"📡 Iniciando polling cada {interval} segundos...")
        
        # Verificar conexión inicial
        if not self.check_connection():
            logger.error("❌ WhatsApp no está conectado en la API")
            logger.error("   Asegúrate de que:")
            logger.error("   1. La API esté corriendo (npm run dev)")
            logger.error("   2. WhatsApp esté autenticado y conectado")
            return
        
        logger.info("✅ WhatsApp conectado. Esperando mensajes...")
        
        try:
            while True:
                # Obtener mensajes nuevos
                messages = self.get_new_messages()
                
                if messages:
                    logger.info(f"📩 {len(messages)} mensaje(s) nuevo(s)")
                    
                    for message in messages:
                        try:
                            # Llamar al callback con el mensaje
                            callback(message)
                        except Exception as e:
                            logger.error(f"Error procesando mensaje: {e}", exc_info=True)
                
                # Esperar antes del siguiente polling
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Deteniendo polling...")
        except Exception as e:
            logger.error(f"Error en polling: {e}", exc_info=True)


# Función auxiliar para extraer número de teléfono del mensaje
def extract_phone_number(message: Dict) -> str:
    """
    Extraer número de teléfono limpio desde mensaje de la API
    
    Args:
        message: Diccionario con datos del mensaje
        
    Returns:
        Número de teléfono limpio
    """
    from_field = message.get('from', '')
    # Remover @c.us
    return from_field.replace('@c.us', '')