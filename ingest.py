"""
Procesador de archivos Markdown a base de conocimiento FAISS
Limpia formato Markdown y crea índice vectorial
MEJORADO: Mejor chunking, limpieza de HTML, más preciso
Genera 2 archivos: faiss_index.bin + knowledge_base.json
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
    """Limpiar formato Markdown y HTML"""
    # Eliminar tags HTML (⭐ NUEVO)
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
    
    # Eliminar código `texto` (⭐ NUEVO)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def parse_markdown_file(filepath):
    """Parsear archivo Markdown dividiéndolo en chunks optimizados"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        filename = filepath.stem
        
        # Dividir por secciones (##) primero
        sections = re.split(r'\n##\s+', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Extraer título de la sección (primera línea)
            lines = section.split('\n', 1)
            title = lines[0].strip() if lines else filename
            body = lines[1] if len(lines) > 1 else section
            
            # ⭐ NUEVO: Dividir sección en chunks más pequeños
            # Dividir por párrafos (doble salto de línea) o por líneas
            paragraphs = re.split(r'\n\n+', body)
            
            # Agrupar párrafos pequeños juntos (max 400 palabras por chunk)
            current_chunk = []
            current_words = 0
            max_words = 450
            
            for para in paragraphs:
                para_clean = clean_markdown(para.strip())
                if not para_clean:
                    continue
                
                para_words = len(para_clean.split())
                
                # Si agregar este párrafo excede el límite, guardar chunk actual
                if current_words + para_words > max_words and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    documents.append({
                        'text': chunk_text,
                        'source': filename,
                        'section': title
                    })
                    current_chunk = [para_clean]
                    current_words = para_words
                else:
                    current_chunk.append(para_clean)
                    current_words += para_words
            
            # Guardar último chunk si existe
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                documents.append({
                    'text': chunk_text,
                    'source': filename,
                    'section': title
                })
        
        return documents
        
    except Exception as e:
        logger.error(f"Error procesando {filepath}: {e}")
        return []


def create_knowledge_base(docs_folder='docs', 
                         index_file='faiss_index.bin',
                         json_file='knowledge_base.json'):
    """Crear base de conocimiento FAISS desde archivos Markdown
    
    Genera 2 archivos:
    - faiss_index.bin: Índice vectorial FAISS
    - knowledge_base.json: Documentos con metadata
    """
    
    logger.info("="*60)
    logger.info("CREANDO BASE DE CONOCIMIENTO")
    logger.info("="*60)
    
    # 1. Cargar modelo de embeddings
    logger.info("Cargando modelo de embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Leer todos los archivos .md
    docs_path = Path(docs_folder)
    if not docs_path.exists():
        logger.error(f"Carpeta {docs_folder} no existe")
        return False
    
    md_files = list(docs_path.glob('*.md'))
    if not md_files:
        logger.error(f"No se encontraron archivos .md en {docs_folder}")
        return False
    
    logger.info(f"Encontrados {len(md_files)} archivos .md")
    
    # 3. Procesar todos los archivos
    all_documents = []
    for md_file in md_files:
        logger.info(f"Procesando: {md_file.name}")
        docs = parse_markdown_file(md_file)
        all_documents.extend(docs)
        logger.info(f"  → {len(docs)} chunks extraídos")
    
    if not all_documents:
        logger.error("No se extrajeron documentos")
        return False
    
    logger.info(f"\nTotal de chunks creados: {len(all_documents)}")
    
    # 4. Generar embeddings
    logger.info("Generando embeddings...")
    texts = [doc['text'] for doc in all_documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 5. Crear índice FAISS
    logger.info("Creando índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # 6. Guardar ÍNDICE FAISS (archivo binario)
    logger.info(f"Guardando índice FAISS en {index_file}...")
    faiss.write_index(index, index_file)
    
    # 7. Guardar DOCUMENTOS (archivo JSON)
    logger.info(f"Guardando documentos en {json_file}...")
    knowledge_data = {
        'documents': all_documents,
        'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'total_docs': len(all_documents),
        'dimension': dimension
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
    
    logger.info("="*60)
    logger.info("✅ BASE DE CONOCIMIENTO CREADA EXITOSAMENTE")
    logger.info(f"   Archivos procesados: {len(md_files)}")
    logger.info(f"   Chunks totales: {len(all_documents)}")
    logger.info(f"   Dimensión embeddings: {dimension}")
    logger.info(f"   Archivo índice: {index_file}")
    logger.info(f"   Archivo JSON: {json_file}")
    logger.info("="*60)
    
    return True


if __name__ == '__main__':
    success = create_knowledge_base()
    exit(0 if success else 1)