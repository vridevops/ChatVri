"""
Utilidades para gestión de la base de conocimiento
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def show_knowledge_base_stats():
    """Mostrar estadísticas de la base de conocimiento"""
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        logger.info("=" * 60)
        logger.info("ESTADÍSTICAS DE LA BASE DE CONOCIMIENTO")
        logger.info("=" * 60)
        logger.info(f"Total de documentos: {len(docs)}")
        
        # Agrupar por fuente
        sources = {}
        for doc in docs:
            source = doc.get('source', 'desconocido')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info(f"\nDocumentos por archivo:")
        for source, count in sorted(sources.items()):
            logger.info(f"  • {source}: {count} secciones")
        
        # Longitud promedio
        lengths = [len(doc['content']) for doc in docs]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        logger.info(f"\nLongitud promedio de contenido: {avg_length:.0f} caracteres")
        logger.info(f"Contenido más corto: {min(lengths)} caracteres")
        logger.info(f"Contenido más largo: {max(lengths)} caracteres")
        
        # Mostrar algunas muestras
        logger.info(f"\nPrimeros 5 documentos:")
        for i, doc in enumerate(docs[:5]):
            logger.info(f"\n  [{i+1}] {doc['title']}")
            logger.info(f"      Fuente: {doc['source']}")
            logger.info(f"      Contenido: {doc['content'][:100]}...")
        
        logger.info("=" * 60)
        
    except FileNotFoundError:
        logger.error("Base de conocimiento no encontrada. Ejecuta: python ingest.py")
    except Exception as e:
        logger.error(f"Error: {e}")


def search_in_knowledge_base(query):
    """Buscar términos en la base de conocimiento (búsqueda simple)"""
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        query_lower = query.lower()
        results = []
        
        for doc in docs:
            if query_lower in doc['content'].lower() or query_lower in doc['title'].lower():
                results.append(doc)
        
        logger.info(f"\nBúsqueda: '{query}'")
        logger.info(f"Resultados encontrados: {len(results)}")
        logger.info("=" * 60)
        
        for i, doc in enumerate(results):
            logger.info(f"\n[{i+1}] {doc['title']}")
            logger.info(f"    Fuente: {doc['source']}")
            
            # Resaltar coincidencias
            content = doc['content']
            if query_lower in content.lower():
                idx = content.lower().find(query_lower)
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 50)
                snippet = "..." + content[start:end] + "..."
                logger.info(f"    {snippet}")
        
        logger.info("=" * 60)
        
    except FileNotFoundError:
        logger.error("Base de conocimiento no encontrada. Ejecuta: python ingest.py")
    except Exception as e:
        logger.error(f"Error: {e}")


def backup_knowledge_base():
    """Crear backup de la base de conocimiento"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path('backups')
        backup_dir.mkdir(exist_ok=True)
        
        # Backup del índice
        if Path('knowledge_base.index').exists():
            import shutil
            shutil.copy('knowledge_base.index', 
                       backup_dir / f'knowledge_base_{timestamp}.index')
            logger.info(f"✓ Backup del índice creado")
        
        # Backup del JSON
        if Path('knowledge_base.json').exists():
            import shutil
            shutil.copy('knowledge_base.json', 
                       backup_dir / f'knowledge_base_{timestamp}.json')
            logger.info(f"✓ Backup del JSON creado")
        
        logger.info(f"\nBackups guardados en: backups/")
        logger.info(f"Timestamp: {timestamp}")
        
    except Exception as e:
        logger.error(f"Error creando backup: {e}")


def list_markdown_files():
    """Listar archivos Markdown disponibles"""
    docs_path = Path('docs')
    
    if not docs_path.exists():
        logger.error("Carpeta docs/ no existe")
        return
    
    md_files = list(docs_path.glob('*.md'))
    
    logger.info("=" * 60)
    logger.info("ARCHIVOS MARKDOWN EN docs/")
    logger.info("=" * 60)
    logger.info(f"Total: {len(md_files)} archivos\n")
    
    for md_file in sorted(md_files):
        size = md_file.stat().st_size
        modified = datetime.fromtimestamp(md_file.stat().st_mtime)
        logger.info(f"  • {md_file.name}")
        logger.info(f"    Tamaño: {size:,} bytes")
        logger.info(f"    Modificado: {modified.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info("=" * 60)


def validate_markdown_structure(filename):
    """Validar estructura de un archivo Markdown"""
    filepath = Path('docs') / filename
    
    if not filepath.exists():
        logger.error(f"Archivo no encontrado: {filepath}")
        return
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Contar secciones ##
        import re
        sections = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
        
        logger.info("=" * 60)
        logger.info(f"VALIDACIÓN: {filename}")
        logger.info("=" * 60)
        logger.info(f"Tamaño total: {len(content):,} caracteres")
        logger.info(f"Secciones encontradas (##): {len(sections)}\n")
        
        if sections:
            logger.info("Secciones:")
            for i, section in enumerate(sections, 1):
                logger.info(f"  {i}. {section}")
        else:
            logger.warning("⚠ No se encontraron secciones con ##")
            logger.warning("  El archivo debe tener encabezados de nivel 2 (##)")
        
        # Buscar patrones comunes
        patterns = {
            'Emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'Teléfonos': r'\b\d{9}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3}\b',
            'Negritas': r'\*\*[^*]+\*\*',
            'Enlaces': r'\[([^\]]+)\]\(([^\)]+)\)'
        }
        
        logger.info("\nPatrones encontrados:")
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, content)
            count = len(matches)
            logger.info(f"  • {pattern_name}: {count}")
            if count > 0 and count <= 3:
                logger.info(f"    Ejemplos: {matches[:3]}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error validando archivo: {e}")


def main():
    """Menú principal"""
    import sys
    
    if len(sys.argv) < 2:
        print("\nUtilidades de gestión de base de conocimiento")
        print("\nUso:")
        print("  python utils.py stats                    - Mostrar estadísticas")
        print("  python utils.py search <término>         - Buscar en base de conocimiento")
        print("  python utils.py backup                   - Crear backup")
        print("  python utils.py list                     - Listar archivos .md")
        print("  python utils.py validate <archivo.md>    - Validar estructura Markdown")
        print()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'stats':
        show_knowledge_base_stats()
    
    elif command == 'search':
        if len(sys.argv) < 3:
            logger.error("Uso: python utils.py search <término>")
        else:
            query = ' '.join(sys.argv[2:])
            search_in_knowledge_base(query)
    
    elif command == 'backup':
        backup_knowledge_base()
    
    elif command == 'list':
        list_markdown_files()
    
    elif command == 'validate':
        if len(sys.argv) < 3:
            logger.error("Uso: python utils.py validate <archivo.md>")
        else:
            validate_markdown_structure(sys.argv[2])
    
    else:
        logger.error(f"Comando desconocido: {command}")


if __name__ == '__main__':
    main()