# Agregar este método a tu clase WhatsAppAPIClient en whatsapp_client.py

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