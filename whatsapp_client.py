
import requests
import time
import logging
from typing import Optional, List, Dict
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class WhatsAppAPIClient:
    """Cliente mejorado para interactuar con la API de WhatsApp"""
    
    def __init__(self, api_url: str, api_key: str):
        """
        Inicializar cliente mejorado
        
        Args:
            api_url: URL base de la API (ej: https://apiwsp.services.vridevops.space)
            api_key: API Key configurada en .env
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.processed_messages = set()
        self.last_check = None
        self.session = requests.Session()  # Session reutilizable
        self.session.headers.update(self._get_headers())
        
    def _get_headers(self) -> dict:
        """Headers comunes para todas las peticiones"""
        return {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key,
            'User-Agent': 'ChatVRI-Bot/1.0'
        }
    
    def send_text(self, to: str, message: str, max_retries: int = 3) -> bool:
        """
        Enviar mensaje de texto CON REINTENTOS
        
        Args:
            to: Número de teléfono (formato: 51987654321)
            message: Mensaje a enviar
            max_retries: Número máximo de reintentos
            
        Returns:
            True si se envió correctamente
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.api_url}/api/whatsapp/send/text"
                payload = {
                    'to': self._format_phone(to),
                    'message': message
                }
                
                # Timeout progresivo: 25s, 35s, 45s
                timeout = 25 + (attempt * 10)
                
                logger.info(f"📤 Intentando enviar a {to} (intento {attempt + 1}/{max_retries}, timeout: {timeout}s)")
                
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Mensaje enviado a {to} (intento {attempt + 1})")
                    return True
                else:
                    logger.warning(f"⚠️ Error HTTP {response.status_code} enviando mensaje: {response.text}")
                    
                    # Si es error 4xx, no reintentar (error del cliente)
                    if 400 <= response.status_code < 500:
                        logger.error(f"❌ Error del cliente {response.status_code}, no se reintenta")
                        return False
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Timeout en intento {attempt + 1} para {to}")
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"🔌 Connection error en intento {attempt + 1}: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error inesperado en intento {attempt + 1}: {e}")
            
            # Backoff exponencial + jitter antes del reintento
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"⏳ Reintentando en {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        logger.error(f"❌ Fallaron todos los {max_retries} intentos para {to}")
        return False
    
    def send_text_async_fallback(self, to: str, message: str) -> None:
        """
        Envío asíncrono que no bloquea el main loop
        Útil para mensajes que pueden fallar sin afectar al usuario
        """
        import threading
        
        def send_in_thread():
            success = self.send_text(to, message)
            if not success:
                logger.error(f"🚨 Fallo crítico enviando a {to} - Mensaje perdido: {message[:50]}...")
        
        thread = threading.Thread(target=send_in_thread, daemon=True)
        thread.start()
    
    def send_image(self, to: str, image_url: str, caption: str = '') -> bool:
        """
        Enviar imagen desde URL CON REINTENTOS
        """
        for attempt in range(3):
            try:
                url = f"{self.api_url}/api/whatsapp/send/media-url"
                payload = {
                    'to': self._format_phone(to),
                    'media': image_url,
                    'caption': caption
                }
                
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return True
                elif 400 <= response.status_code < 500:
                    return False  # No reintentar errores del cliente
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout enviando imagen (intento {attempt + 1})")
            except Exception as e:
                logger.error(f"Error enviando imagen: {e}")
            
            if attempt < 2:
                time.sleep(2 ** attempt)
        
        return False
    
    def get_new_messages(self, limit: int = 50) -> List[Dict]:
        """
        Obtener mensajes nuevos con manejo robusto de errores
        """
        try:
            url = f"{self.api_url}/api/whatsapp/messages"
            params = {'limit': limit}
            
            response = self.session.get(
                url,
                params=params,
                timeout=15  # Timeout reducido para polling
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
                
                # Solo mensajes de usuarios (terminan en @c.us) y que no hemos procesado
                if '@c.us' in from_user and msg_id not in self.processed_messages:
                    self.processed_messages.add(msg_id)
                    
                    # Asegurar formato consistente del mensaje
                    msg['body'] = msg.get('body', '').strip()
                    msg['from_clean'] = self._format_phone(from_user)
                    
                    new_messages.append(msg)
            
            # Limpiar mensajes procesados antiguos (mantener últimos 1000)
            if len(self.processed_messages) > 1000:
                self.processed_messages = set(list(self.processed_messages)[-1000:])
            
            return new_messages
            
        except requests.exceptions.Timeout:
            logger.warning("Timeout obteniendo mensajes")
            return []
        except Exception as e:
            logger.error(f"Error en get_new_messages: {e}")
            return []
    
    def check_connection(self) -> bool:
        """
        Verificar si WhatsApp está conectado con timeout corto
        """
        try:
            url = f"{self.api_url}/api/whatsapp/status"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                connected = data.get('data', {}).get('connected', False)
                if connected:
                    logger.info("✅ WhatsApp conectado")
                else:
                    logger.warning("⚠️ WhatsApp desconectado en la API")
                return connected
            
            return False
            
        except requests.exceptions.Timeout:
            logger.warning("Timeout verificando conexión WhatsApp")
            return False
        except Exception as e:
            logger.error(f"Error verificando conexión: {e}")
            return False
    
    def health_check(self) -> Dict:
        """
        Verificación completa de salud del servicio
        """
        health_info = {
            'whatsapp_connected': False,
            'api_accessible': False,
            'response_time': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            url = f"{self.api_url}/api/whatsapp/status"
            response = self.session.get(url, timeout=10)
            health_info['response_time'] = time.time() - start_time
            
            if response.status_code == 200:
                health_info['api_accessible'] = True
                data = response.json()
                health_info['whatsapp_connected'] = data.get('data', {}).get('connected', False)
            else:
                health_info['error'] = f"HTTP {response.status_code}"
                
        except Exception as e:
            health_info['error'] = str(e)
        
        return health_info
    
    def _format_phone(self, phone: str) -> str:
        """
        Formatear número de teléfono robusto
        """
        if not phone:
            return ""
        
        # Remover @c.us si existe
        if '@c.us' in phone:
            phone = phone.replace('@c.us', '')
        
        # Remover espacios y caracteres especiales, mantener solo dígitos
        phone = ''.join(filter(str.isdigit, phone))
        
        # Asegurar que empiece con 51 (Perú)
        if phone.startswith('51') and len(phone) == 11:
            return phone
        elif len(phone) == 9:  # Si solo tiene 9 dígitos, agregar 51
            return '51' + phone
        else:
            return phone  # Devolver como está si no cumple patrones
    
    def start_polling(self, callback, interval: int = 2):
        """
        Polling robusto con manejo de errores continuo
        """
        logger.info(f"📡 Iniciando polling cada {interval} segundos...")
        
        # Verificar conexión inicial
        health = self.health_check()
        if not health['api_accessible']:
            logger.error("❌ No se puede acceder a la API de WhatsApp")
            logger.error(f"   Error: {health.get('error', 'Desconocido')}")
            return
        
        if not health['whatsapp_connected']:
            logger.warning("⚠️ WhatsApp no está conectado en la API")
            logger.warning("   Los mensajes se encolarán pero no se enviarán hasta la conexión")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                try:
                    # Verificar salud periódicamente
                    if consecutive_errors > 0 or time.time() % 60 < interval:
                        health = self.health_check()
                        if not health['api_accessible']:
                            consecutive_errors += 1
                            logger.error(f"❌ API inaccesible (error {consecutive_errors}/{max_consecutive_errors})")
                            if consecutive_errors >= max_consecutive_errors:
                                logger.error("🚨 Demasiados errores consecutivos, reiniciando polling...")
                                break
                        else:
                            consecutive_errors = 0
                    
                    # Obtener mensajes nuevos
                    messages = self.get_new_messages()
                    
                    if messages:
                        logger.info(f"📩 {len(messages)} mensaje(s) nuevo(s)")
                        
                        for message in messages:
                            try:
                                callback(message)
                            except Exception as e:
                                logger.error(f"Error en callback para mensaje: {e}", exc_info=True)
                    
                    # Esperar antes del siguiente polling
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error en ciclo de polling: {e}")
                    consecutive_errors += 1
                    time.sleep(interval * 2)  # Backoff en caso de error
                
        except KeyboardInterrupt:
            logger.info("\n👋 Deteniendo polling...")
        finally:
            self.session.close()
            logger.info("✅ Sesión HTTP cerrada")


# Función auxiliar para extraer número de teléfono del mensaje
def extract_phone_number(message: Dict) -> str:
    """
    Extraer número de teléfono limpio desde mensaje de la API
    """
    from_field = message.get('from', '')
    from_clean = message.get('from_clean', '')
    
    if from_clean:
        return from_clean
    
    # Fallback: limpiar el campo from
    return from_field.replace('@c.us', '')