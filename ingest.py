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
    """Limpiar formato Markdown manteniendo estructura"""
    # Eliminar tags HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Eliminar enlaces pero mantener texto
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Eliminar formato markdown básico
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_standard_markdown(filepath):
    """Procesador UNIVERSAL para formato estándar UNA Puno"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        filename = filepath.stem
        
        # ⭐ EXTRAER METADATOS YAML
        yaml_metadata = {}
        yaml_match = re.search(r'^---\n(.+?)\n---', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    yaml_metadata[key.strip()] = value.strip()
            # Remover YAML del contenido
            content = content[yaml_match.end():].strip()
        
        file_type = yaml_metadata.get('TIPO', 'general')
        entidad = yaml_metadata.get('ENTIDAD', 'general')
        
        # ⭐ PROCESAMIENTO UNIVERSAL por secciones ##
        sections = re.split(r'\n##\s+', content)
        
        current_facultad = ""
        current_escuela = ""
        
        for section in sections:
            if not section.strip():
                continue
            
            # Extraer título principal
            title_match = re.match(r'(.+?)(?:\n|$)', section)
            if not title_match:
                continue
                
            title = title_match.group(1).strip()
            body = section[len(title):].strip()
            
            # ⭐ DETECTAR TIPO DE SECCIÓN por patrones
            if title.startswith('FACULTAD_'):
                current_facultad = title
                current_escuela = ""
                continue
            elif title.startswith('ESCUELA_') or 'INGENIERIA_' in title or 'MEDICINA_' in title:
                current_escuela = title
                continue
            elif 'ETAPA_' in title or 'PROCESO_' in title:
                # Es una etapa de proceso
                process_doc = create_process_document(title, body, filename, file_type, entidad)
                if process_doc:
                    documents.append(process_doc)
                continue
            elif any(keyword in title.upper() for keyword in ['INTRODUCCION', 'ARTICULO', 'REGLAMENTO', 'NORMAS']):
                # Es artículo de reglamento
                reglamento_doc = create_reglamento_document(title, body, filename, file_type, entidad)
                if reglamento_doc:
                    documents.append(reglamento_doc)
                continue
            
            # ⭐ PROCESAR SECCIÓN CON CAMPOS ESTANDARIZADOS
            section_data = extract_standard_fields(body)
            
            if section_data:
                # Crear documento estructurado
                doc = {
                    'text': create_structured_text(title, section_data, file_type),
                    'source': filename,
                    'type': file_type,
                    'entidad': entidad,
                    'facultad': current_facultad or section_data.get('facultad', ''),
                    'escuela': current_escuela or section_data.get('escuela', ''),
                    'metadata': section_data
                }
                documents.append(doc)
        
        logger.info(f"📄 {filename}: {len(documents)} documentos creados")
        return documents
        
    except Exception as e:
        logger.error(f"Error procesando {filepath}: {e}")
        return []

def extract_standard_fields(body):
    """Extraer campos estandarizados **CAMPO:** valor"""
    fields = {}
    
    # Patrón para campos estandarizados
    field_pattern = r'\*\*([A-Z_]+):\*\*\s*(.+?)(?=\n\*\*|\n##|\n\n|$)'
    matches = re.findall(field_pattern, body, re.DOTALL | re.IGNORECASE)
    
    for field_name, field_value in matches:
        clean_value = clean_markdown(field_value.strip())
        fields[field_name.lower()] = clean_value
    
    # Si no hay campos estandarizados, usar el contenido completo
    if not fields and body.strip():
        fields['contenido'] = clean_markdown(body)
    
    return fields

def create_structured_text(title, fields, file_type):
    """Crear texto estructurado para embeddings"""
    text_parts = [f"TITULO: {title}"]
    
    # Campos prioritarios según tipo
    priority_fields = {
        'coordinadores': ['nombre', 'email', 'telefono', 'horario', 'ubicacion'],
        'lineas_investigacion': ['linea', 'sublineas', 'descripcion'],
        'procesos': ['descripcion', 'plazo', 'responsable', 'requisitos'],
        'reglamento': ['contenido', 'articulo', 'norma'],
        'preguntas_frecuentes': ['pregunta', 'respuesta']
    }
    
    # Agregar campos prioritarios primero
    for field in priority_fields.get(file_type, []):
        if field in fields:
            text_parts.append(f"{field.upper()}: {fields[field]}")
    
    # Agregar todos los demás campos
    for field, value in fields.items():
        if field not in priority_fields.get(file_type, []):
            text_parts.append(f"{field.upper()}: {value}")
    
    return "\n".join(text_parts)

def create_process_document(title, body, filename, file_type, entidad):
    """Crear documento para etapas de proceso"""
    fields = extract_standard_fields(body)
    if not fields:
        fields['descripcion'] = clean_markdown(body)
    
    return {
        'text': create_structured_text(title, fields, file_type),
        'source': filename,
        'type': 'etapa_proceso',
        'entidad': entidad,
        'etapa': title,
        'metadata': fields
    }

def create_reglamento_document(title, body, filename, file_type, entidad):
    """Crear documento para artículos de reglamento"""
    fields = extract_standard_fields(body)
    if not fields:
        fields['contenido'] = clean_markdown(body)
    
    return {
        'text': create_structured_text(title, fields, file_type),
        'source': filename,
        'type': 'articulo_reglamento',
        'entidad': entidad,
        'articulo': title,
        'metadata': fields
    }

def create_knowledge_base(docs_folder='docs', 
                         index_file='faiss_index.bin',
                         json_file='knowledge_base.json'):
    """Crear base de conocimiento FAISS - VERSIÓN SIMPLIFICADA"""
    
    logger.info("="*60)
    logger.info("CREANDO BASE DE CONOCIMIENTO UNA PUNO - FORMATO ESTÁNDAR")
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
    
    # 3. Procesar todos los archivos con el PROCESADOR UNIVERSAL
    all_documents = []
    for md_file in md_files:
        logger.info(f"Procesando: {md_file.name}")
        docs = parse_standard_markdown(md_file)
        all_documents.extend(docs)
        logger.info(f"  → {len(docs)} documentos estructurados")
    
    if not all_documents:
        logger.error("No se extrajeron documentos")
        return False
    
    # Estadísticas
    type_counts = {}
    for doc in all_documents:
        doc_type = doc.get('type', 'general')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    logger.info(f"\n📊 ESTADÍSTICAS:")
    for doc_type, count in type_counts.items():
        logger.info(f"   {doc_type}: {count}")
    logger.info(f"   TOTAL: {len(all_documents)}")
    
    # 4. Generar embeddings
    logger.info("Generando embeddings...")
    texts = [doc['text'] for doc in all_documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 5. Crear índice FAISS
    logger.info("Creando índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # 6. Guardar archivos
    logger.info(f"Guardando índice FAISS...")
    faiss.write_index(index, index_file)
    
    logger.info(f"Guardando documentos JSON...")
    knowledge_data = {
        'documents': all_documents,
        'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'total_docs': len(all_documents),
        'dimension': dimension,
        'chunk_types': type_counts,
        'created_at': str(np.datetime64('now'))
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
    
    logger.info("="*60)
    logger.info("✅ BASE DE CONOCIMIENTO CREADA EXITOSAMENTE")
    logger.info(f"   Archivos: {len(md_files)}")
    logger.info(f"   Documentos: {len(all_documents)}")
    logger.info("="*60)
    
    return True

if __name__ == '__main__':
    success = create_knowledge_base()
    exit(0 if success else 1)