import os
import requests
import time
import logging
from typing import List, Dict, Set
from datetime import datetime

logger = logging.getLogger('whatsapp_client')

class WhatsAppClient:
    def __init__(self):
        self.base_url = os.environ.get('WHATSAPP_API_URL', 'https://apiwsp.services.vridevops.space')
        self.api_key = os.environ.get('WHATSAPP_API_KEY', '')
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': self.api_key})
        
        # NUEVO: Conjunto para rastrear mensajes ya procesados
        self.processed_message_ids: Set[str] = set()
        
        # NUEVO: Límite de mensajes en memoria (evitar crecimiento infinito)
        self.max_processed_ids = 1000
        
        # NUEVO: Timeout más corto para detección rápida de problemas
        self.timeout = 10
        
        # NUEVO: Circuit breaker para evitar spam cuando la API está caída
        self.consecutive_failures = 0
        self.max_failures_before_backoff = 3
        self.backoff_time = 60  # segundos

    def _create_message_id(self, message: Dict) -> str:
        """
        Crear ID único para cada mensaje basado en:
        - Número de teléfono
        - Timestamp
        - Contenido del mensaje (hash)
        """
        phone = message.get('from', '')
        timestamp = message.get('timestamp', '')
        body = message.get('body', '')
        
        # Crear ID único combinando datos
        message_id = f"{phone}:{timestamp}:{hash(body)}"
        return message_id

    def _cleanup_processed_ids(self):
        """Limpiar IDs antiguos si excedemos el límite"""
        if len(self.processed_message_ids) > self.max_processed_ids:
            # Mantener solo los últimos 500 IDs
            self.processed_message_ids = set(list(self.processed_message_ids)[-500:])
            logger.info(f"🧹 Limpieza de IDs: manteniendo últimos 500")

    def get_messages(self, limit: int = 50, unread_only: bool = True) -> List[Dict]:
        """
        Obtener mensajes nuevos con protección contra duplicados
        """
        try:
            # Circuit breaker: si hay muchos fallos, esperar antes de reintentar
            if self.consecutive_failures >= self.max_failures_before_backoff:
                logger.warning(f"⚠️ Circuit breaker activo. Esperando {self.backoff_time}s...")
                time.sleep(self.backoff_time)
                self.consecutive_failures = 0

            url = f"{self.base_url}/api/whatsapp/messages"
            params = {
                'limit': limit,
                'unreadOnly': unread_only
            }
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout,
                verify=True  # Verificar SSL
            )
            
            # Verificar código de respuesta
            if response.status_code == 200:
                self.consecutive_failures = 0  # Reset en éxito
                messages = response.json()
                
                if not messages:
                    return []
                
                logger.info(f"📬 Recibidos {len(messages)} mensajes")
                
                # NUEVO: Filtrar mensajes ya procesados
                new_messages = []
                for msg in messages:
                    msg_id = self._create_message_id(msg)
                    
                    if msg_id not in self.processed_message_ids:
                        new_messages.append(msg)
                        self.processed_message_ids.add(msg_id)
                    else:
                        logger.debug(f"⏭️ Mensaje duplicado ignorado: {msg.get('from', 'unknown')}")
                
                # Limpiar IDs antiguos si es necesario
                self._cleanup_processed_ids()
                
                if new_messages:
                    logger.info(f"📬 {len(new_messages)} mensajes nuevos (filtrados {len(messages) - len(new_messages)} duplicados)")
                else:
                    logger.debug(f"⏭️ Todos los {len(messages)} mensajes ya fueron procesados")
                
                return new_messages
                
            elif response.status_code == 401:
                logger.error("❌ API Key inválida o expirada")
                self.consecutive_failures += 1
                return []
                
            elif response.status_code == 503:
                logger.warning("⚠️ API temporalmente no disponible (503)")
                self.consecutive_failures += 1
                return []
                
            else:
                logger.error(f"❌ Error HTTP {response.status_code}: {response.text}")
                self.consecutive_failures += 1
                return []
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Timeout al conectar con API (>{self.timeout}s)")
            self.consecutive_failures += 1
            return []
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Error de conexión: {str(e)[:100]}")
            self.consecutive_failures += 1
            
            # Solo mostrar detalle en el primer fallo
            if self.consecutive_failures == 1:
                logger.error(f"🔍 Verifica que la API esté corriendo en: {self.base_url}")
            
            return []
            
        except requests.exceptions.SSLError as e:
            logger.error(f"🔒 Error SSL: {str(e)[:100]}")
            self.consecutive_failures += 1
            return []
            
        except Exception as e:
            logger.error(f"❌ Error inesperado: {type(e).__name__}: {str(e)[:100]}")
            self.consecutive_failures += 1
            return []

    def send_message(self, to: str, message: str) -> bool:
        """
        Enviar mensaje con mejor manejo de errores
        """
        try:
            url = f"{self.base_url}/api/whatsapp/send"
            data = {
                'to': to,
                'message': message
            }
            
            response = self.session.post(
                url, 
                json=data, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Mensaje enviado a {to}")
                return True
            elif response.status_code == 429:
                logger.warning(f"⚠️ Rate limit excedido al enviar a {to}")
                return False
            else:
                logger.error(f"❌ Error al enviar mensaje ({response.status_code}): {response.text[:100]}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Timeout al enviar mensaje a {to}")
            return False
            
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ No se pudo conectar a API para enviar mensaje a {to}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error al enviar mensaje: {type(e).__name__}: {str(e)[:100]}")
            return False

    def mark_as_read(self, message_id: str) -> bool:
        """
        NUEVO: Marcar mensaje como leído explícitamente
        """
        try:
            url = f"{self.base_url}/api/whatsapp/messages/{message_id}/read"
            response = self.session.post(url, timeout=5)
            
            if response.status_code == 200:
                logger.debug(f"✓ Mensaje {message_id} marcado como leído")
                return True
            else:
                logger.warning(f"⚠️ No se pudo marcar mensaje {message_id} como leído")
                return False
                
        except Exception as e:
            logger.debug(f"Error al marcar como leído: {e}")
            return False

    def get_connection_status(self) -> Dict:
        """
        Verificar estado de conexión de la API
        """
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                return {'status': 'connected', 'details': response.json()}
            else:
                return {'status': 'error', 'code': response.status_code}
                
        except Exception as e:
            return {'status': 'disconnected', 'error': str(e)}

    def reset_circuit_breaker(self):
        """Resetear circuit breaker manualmente"""
        self.consecutive_failures = 0
        logger.info("🔄 Circuit breaker reseteado")


# Instancia global
whatsapp_client = WhatsAppClient()


def get_messages(limit: int = 50, unread_only: bool = True) -> List[Dict]:
    """Wrapper para compatibilidad con código existente"""
    return whatsapp_client.get_messages(limit, unread_only)


def send_message(to: str, message: str) -> bool:
    """Wrapper para compatibilidad con código existente"""
    return whatsapp_client.send_message(to, message)


def get_connection_status() -> Dict:
    """Wrapper para obtener estado de conexión"""
    return whatsapp_client.get_connection_status()