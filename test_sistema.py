"""
Tests de verificación del sistema completo
"""

import os
import sys
import json
import logging
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_files_exist():
    """Verificar que existan todos los archivos necesarios"""
    logger.info("\n=== TEST 1: Verificando archivos ===")
    
    required_files = [
        'main.py',
        'ingest.py',
        'requirements.txt',
        '.env',
        'utils.py'
    ]
    
    all_exist = True
    for filename in required_files:
        exists = Path(filename).exists()
        status = "✓" if exists else "✗"
        logger.info(f"{status} {filename}")
        if not exists:
            all_exist = False
    
    # Verificar carpeta docs
    docs_exist = Path('docs').exists()
    status = "✓" if docs_exist else "✗"
    logger.info(f"{status} docs/")
    if not docs_exist:
        all_exist = False
    
    return all_exist


def test_env_variables():
    """Verificar variables de entorno"""
    logger.info("\n=== TEST 2: Verificando variables de entorno ===")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'TWILIO_ACCOUNT_SID',
        'TWILIO_AUTH_TOKEN',
        'TWILIO_WHATSAPP_NUMBER',
        'OLLAMA_URL',
        'OLLAMA_MODEL',
        'OLLAMA_MODEL_BACKUP'
    ]
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        is_set = value and value != f'tu_{var.lower()}_aqui'
        status = "✓" if is_set else "✗"
        display_value = value[:20] + "..." if value and len(value) > 20 else value
        logger.info(f"{status} {var}: {display_value}")
        if not is_set:
            all_set = False
    
    return all_set


def test_ollama_connection():
    """Verificar conexión con Ollama"""
    logger.info("\n=== TEST 3: Verificando conexión Ollama ===")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✓ Conectado a Ollama en {ollama_url}")
            logger.info(f"  Modelos disponibles: {len(models)}")
            
            # Verificar modelos requeridos
            model_names = [m['name'] for m in models]
            required_models = [
                os.getenv('OLLAMA_MODEL'),
                os.getenv('OLLAMA_MODEL_BACKUP')
            ]
            
            for model in required_models:
                if any(model in name for name in model_names):
                    logger.info(f"  ✓ {model}")
                else:
                    logger.warning(f"  ✗ {model} NO ENCONTRADO")
                    logger.warning(f"    Ejecuta: ollama pull {model}")
            
            return True
        else:
            logger.error(f"✗ Error conectando a Ollama: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ No se pudo conectar a Ollama: {e}")
        logger.error(f"  Asegúrate de ejecutar: ollama serve")
        return False


def test_knowledge_base():
    """Verificar base de conocimiento"""
    logger.info("\n=== TEST 4: Verificando base de conocimiento ===")
    
    index_exists = Path('knowledge_base.index').exists()
    json_exists = Path('knowledge_base.json').exists()
    
    if not index_exists or not json_exists:
        logger.warning("✗ Base de conocimiento no encontrada")
        logger.warning("  Ejecuta: python ingest.py")
        return False
    
    logger.info("✓ knowledge_base.index")
    logger.info("✓ knowledge_base.json")
    
    # Verificar contenido
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
        logger.info(f"  → {len(docs)} documentos indexados")
        
        if len(docs) > 0:
            logger.info(f"  Ejemplo: {docs[0]['title']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error leyendo base de conocimiento: {e}")
        return False


def test_dependencies():
    """Verificar dependencias instaladas"""
    logger.info("\n=== TEST 5: Verificando dependencias ===")
    
    dependencies = [
        'flask',
        'twilio',
        'dotenv',
        'requests',
        'faiss',
        'sentence_transformers',
        'numpy'
    ]
    
    all_installed = True
    for dep in dependencies:
        try:
            if dep == 'dotenv':
                __import__('dotenv')
            elif dep == 'faiss':
                __import__('faiss')
            else:
                __import__(dep)
            logger.info(f"✓ {dep}")
        except ImportError:
            logger.error(f"✗ {dep} NO INSTALADO")
            all_installed = False
    
    if not all_installed:
        logger.warning("\n  Instala las dependencias con:")
        logger.warning("  pip install -r requirements.txt")
    
    return all_installed


def test_markdown_docs():
    """Verificar archivos Markdown en docs/"""
    logger.info("\n=== TEST 6: Verificando archivos Markdown ===")
    
    docs_path = Path('docs')
    if not docs_path.exists():
        logger.error("✗ Carpeta docs/ no existe")
        return False
    
    md_files = list(docs_path.glob('*.md'))
    
    if len(md_files) == 0:
        logger.warning("✗ No se encontraron archivos .md en docs/")
        return False
    
    logger.info(f"✓ Encontrados {len(md_files)} archivos .md:")
    for md_file in md_files:
        size = md_file.stat().st_size
        logger.info(f"  • {md_file.name} ({size} bytes)")
    
    return True


def run_all_tests():
    """Ejecutar todos los tests"""
    logger.info("=" * 60)
    logger.info("VERIFICACIÓN DEL SISTEMA - CHATBOT WHATSAPP UNA PUNO")
    logger.info("=" * 60)
    
    results = {
        'Archivos': test_files_exist(),
        'Variables de entorno': test_env_variables(),
        'Conexión Ollama': test_ollama_connection(),
        'Base de conocimiento': test_knowledge_base(),
        'Dependencias': test_dependencies(),
        'Archivos Markdown': test_markdown_docs()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE TESTS")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("\n✓✓✓ TODOS LOS TESTS PASARON ✓✓✓")
        logger.info("\nEl sistema está listo para usar.")
        logger.info("Ejecuta: python main.py")
    else:
        logger.warning("\n⚠ ALGUNOS TESTS FALLARON ⚠")
        logger.warning("\nRevisa los errores arriba y corrígelos antes de continuar.")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)