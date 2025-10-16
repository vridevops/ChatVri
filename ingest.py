"""
Procesador de archivos Markdown a base de conocimiento FAISS
Limpia formato Markdown y crea índice vectorial
MEJORADO: Soporta HTML, frontmatter, mejor chunking
"""

import os
import json
import re
import logging
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_markdown(text):
    """Limpiar formato Markdown Y HTML"""
    # ⭐ NUEVO: Eliminar tags HTML
    text = re.sub(r'<ul>|</ul>|<li>|</li>|<br/?>|</br>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Eliminar enlaces [texto](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Eliminar mailto: y tel:
    text = re.sub(r'mailto:|tel:', '', text)
    
    # Eliminar negritas **texto**
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    
    # Eliminar cursivas *texto*
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # ⭐ NUEVO: Eliminar código `texto`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_frontmatter(content):
    """⭐ NUEVO: Extraer metadata YAML si existe"""
    metadata = {}
    pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        frontmatter = match.group(1)
        content = content[match.end():]
        
        # Parsear campos simples
        for line in frontmatter.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip().lower()] = value.strip()
    
    return metadata, content


def parse_markdown_file(filepath):
    """Parsear archivo Markdown por secciones (encabezados ##)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ⭐ NUEVO: Extraer frontmatter
        metadata_base, content = extract_frontmatter(content)
        
        # Dividir por secciones ##
        sections = re.split(r'\n##\s+', content)
        
        documents = []
        for section in sections:
            if not section.strip():
                continue
            
            # Primera línea es el título
            lines = section.split('\n', 1)
            if len(lines) < 2:
                continue
            
            title = lines[0].strip()
            content_text = lines[1].strip() if len(lines) > 1 else ""
            
            # Limpiar formato
            cleaned_content = clean_markdown(content_text)
            
            if cleaned_content:
                doc = {
                    'title': title,
                    'content': cleaned_content,
                    'source': filepath.name
                }
                
                # ⭐ NUEVO: Agregar metadata del frontmatter si existe
                if metadata_base:
                    doc['metadata'] = metadata_base
                
                documents.append(doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error parseando {filepath}: {e}")
        return []


def process_documents_folder(docs_folder='docs'):
    """Procesar todos los archivos .md de la carpeta docs/"""
    docs_path = Path(docs_folder)
    
    if not docs_path.exists():
        logger.error(f"Carpeta {docs_folder}/ no encontrada")
        return []
    
    all_documents = []
    md_files = list(docs_path.glob('*.md'))
    
    logger.info(f"Encontrados {len(md_files)} archivos .md")
    
    for md_file in md_files:
        logger.info(f"Procesando: {md_file.name}")
        docs = parse_markdown_file(md_file)
        all_documents.extend(docs)
        logger.info(f"  → {len(docs)} secciones extraídas")
    
    return all_documents


def create_faiss_index(documents, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """Crear índice FAISS con embeddings"""
    try:
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info(f"Generando embeddings para {len(documents)} documentos...")
        texts = [doc['content'] for doc in documents]
        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        logger.info("Creando índice FAISS...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logger.info(f"✓ Índice FAISS creado: {index.ntotal} vectores de dimensión {dimension}")
        return index
        
    except Exception as e:
        logger.error(f"Error creando índice FAISS: {e}")
        return None


def save_knowledge_base(index, documents):
    """Guardar índice FAISS y documentos"""
    try:
        # Guardar índice FAISS
        faiss.write_index(index, 'knowledge_base.index')
        logger.info("✓ Índice FAISS guardado: knowledge_base.index")
        
        # Guardar documentos
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logger.info("✓ Documentos guardados: knowledge_base.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Error guardando base de conocimiento: {e}")
        return False


def main():
    """Proceso principal"""
    logger.info("=" * 60)
    logger.info("PROCESADOR DE BASE DE CONOCIMIENTO")
    logger.info("=" * 60)
    
    # Procesar archivos Markdown
    logger.info("\n1. Procesando archivos Markdown...")
    documents = process_documents_folder('docs')
    
    if not documents:
        logger.error("No se encontraron documentos para procesar")
        return
    
    logger.info(f"✓ Total de documentos procesados: {len(documents)}")
    
    # Mostrar muestra
    logger.info("\nMuestra de documentos procesados:")
    for i, doc in enumerate(documents[:3]):
        logger.info(f"\n  [{i+1}] {doc['title']}")
        logger.info(f"      Contenido: {doc['content'][:100]}...")
        logger.info(f"      Fuente: {doc['source']}")
        if 'metadata' in doc:
            logger.info(f"      Metadata: {doc['metadata']}")
    
    # Crear índice FAISS
    logger.info("\n2. Creando índice FAISS...")
    index = create_faiss_index(documents)
    
    if not index:
        logger.error("Error creando índice FAISS")
        return
    
    # Guardar base de conocimiento
    logger.info("\n3. Guardando base de conocimiento...")
    if save_knowledge_base(index, documents):
        logger.info("\n" + "=" * 60)
        logger.info("✓ BASE DE CONOCIMIENTO CREADA EXITOSAMENTE")
        logger.info("=" * 60)
        logger.info("\nArchivos generados:")
        logger.info("  • knowledge_base.index (índice FAISS)")
        logger.info("  • knowledge_base.json (documentos)")
        logger.info(f"\nTotal de documentos indexados: {len(documents)}")
        logger.info("\nAhora puedes ejecutar: python main.py")
    else:
        logger.error("Error guardando base de conocimiento")


if __name__ == '__main__':
    main()