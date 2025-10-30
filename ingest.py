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
    """Limpiar formato Markdown y HTML - PRESERVAR datos importantes"""
    # Eliminar tags HTML pero preservar contenido
    text = re.sub(r'<ul>|</ul>|<li>|</li>|<br/?>|</br>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Eliminar enlaces pero preservar texto
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Eliminar mailto: y tel: pero preservar números
    text = re.sub(r'mailto:|tel:', '', text)
    
    # ⭐ MODIFICADO: Preservar negritas para datos importantes
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Pero quitar los **
    
    # Eliminar cursivas
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # Eliminar código inline
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_una_puno_markdown(filepath):
    """Procesador OPTIMIZADO para estructura UNA Puno"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        filename = filepath.stem
        
        # ⭐ ESTRATEGIA ESPECÍFICA POR TIPO DE ARCHIVO
        if 'coordinador' in filename.lower():
            return process_coordinadores_file(content, filename)
        elif 'lineas' in filename.lower() or 'sublineas' in filename.lower():
            return process_lineas_investigacion_file(content, filename)
        elif 'preguntas' in filename.lower() or 'faq' in filename.lower():
            return process_preguntas_frecuentes_file(content, filename)
        elif 'proceso' in filename.lower() or 'mapa' in filename.lower():
            return process_procesos_file(content, filename)
        elif 'reglamento' in filename.lower():
            return process_reglamento_file(content, filename)
        else:
            return process_generic_file(content, filename)
            
    except Exception as e:
        logger.error(f"Error procesando {filepath}: {e}")
        return []

def process_coordinadores_file(content, filename):
    """Procesar archivo de coordinadores - UNA ESTRUCTURA POR FACULTAD"""
    documents = []
    
    # Dividir por facultades (separadas por ---)
    sections = re.split(r'\n---+\n', content)
    
    for section in sections:
        if not section.strip() or not section.startswith('##'):
            continue
        
        # Extraer nombre de facultad
        faculty_match = re.search(r'##\s+(.+?)\n', section)
        if not faculty_match:
            continue
            
        faculty = faculty_match.group(1).strip()
        
        # ⭐ CREAR UN SOLO CHUNK POR FACULTAD (más cohesivo)
        clean_content = clean_markdown(section)
        
        # Extraer datos estructurados
        coordinador_match = re.search(r'Coordinadora?:\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
        email_match = re.search(r'Email:\s*([^\n]+)', section, re.IGNORECASE)
        telefono_match = re.search(r'Teléfono:\s*([^\n]+)', section, re.IGNORECASE)
        horario_match = re.search(r'Horario:\s*([^\n]+)', section, re.IGNORECASE)
        ubicacion_match = re.search(r'Ubicación:\s*([^\n]+)', section, re.IGNORECASE)
        
        # Construir texto enriquecido para mejores embeddings
        enriched_text = f"FACULTAD: {faculty}\n"
        if coordinador_match:
            enriched_text += f"COORDINADOR: {coordinador_match.group(1).strip()}\n"
        if email_match:
            enriched_text += f"EMAIL: {email_match.group(1).strip()}\n"
        if telefono_match:
            enriched_text += f"TELÉFONO: {telefono_match.group(1).strip()}\n"
        if horario_match:
            enriched_text += f"HORARIO: {horario_match.group(1).strip()}\n"
        if ubicacion_match:
            enriched_text += f"UBICACIÓN: {ubicacion_match.group(1).strip()}\n"
        
        enriched_text += f"\nINFORMACIÓN COMPLETA:\n{clean_content}"
        
        documents.append({
            'text': enriched_text,
            'source': filename,
            'facultad': faculty,
            'type': 'coordinador',
            'coordinador': coordinador_match.group(1).strip() if coordinador_match else '',
            'email': email_match.group(1).strip() if email_match else '',
            'telefono': telefono_match.group(1).strip() if telefono_match else ''
        })
    
    logger.info(f"📞 Coordinadores procesados: {len(documents)} facultades")
    return documents

def process_lineas_investigacion_file(content, filename):
    """Procesar archivo de líneas de investigación"""
    documents = []
    
    # Dividir por secciones principales
    sections = re.split(r'\n---+\n', content)
    
    current_faculty = ""
    for section in sections:
        if not section.strip():
            continue
        
        # Detectar facultades (## FACULTAD DE ...)
        faculty_match = re.search(r'##\s+FACULTAD DE (.+?)\n', section, re.IGNORECASE)
        if faculty_match:
            current_faculty = faculty_match.group(1).strip()
            continue
        
        if not current_faculty:
            continue
            
        # Detectar escuelas profesionales (### ... Escuela Profesional de ...)
        school_matches = re.finditer(r'###\s+.+?Escuela Profesional de (.+?)\n', section, re.IGNORECASE)
        
        for school_match in school_matches:
            school = school_match.group(1).strip()
            
            # Extraer líneas de investigación
            line_match = re.search(r'\*\*Línea de Investigación:\*\*(.+?)(?=\n\n|\n\*\*|\n###|$)', section, re.DOTALL)
            sublines_match = re.search(r'\*\*Sublíneas de Investigación:\*\*(.+?)(?=\n\n|\n\*\*|\n###|$)', section, re.DOTALL)
            
            if line_match:
                linea_text = clean_markdown(line_match.group(1).strip())
                
                documents.append({
                    'text': f"FACULTAD: {current_faculty}\nESCUELA: {school}\nLÍNEA DE INVESTIGACIÓN: {linea_text}",
                    'source': filename,
                    'facultad': current_faculty,
                    'escuela': school,
                    'type': 'linea_investigacion',
                    'linea': linea_text
                })
            
            if sublines_match:
                sublines_text = clean_markdown(sublines_match.group(1).strip())
                
                documents.append({
                    'text': f"FACULTAD: {current_faculty}\nESCUELA: {school}\nSUBÍNEAS DE INVESTIGACIÓN:\n{sublines_text}",
                    'source': filename,
                    'facultad': current_faculty,
                    'escuela': school,
                    'type': 'sublinea_investigacion',
                    'sublines': sublines_text
                })
    
    logger.info(f"🔬 Líneas investigación procesadas: {len(documents)} entradas")
    return documents

def process_preguntas_frecuentes_file(content, filename):
    """Procesar archivo de preguntas frecuentes"""
    documents = []
    
    # Procesar tablas de preguntas/respuestas
    table_matches = re.findall(r'\|\s*(.+?)\s*\|\s*(.+?)\s*\|', content)
    
    for question, answer in table_matches:
        if question.strip() and not question.startswith('---') and not question.startswith('Pregunta'):
            clean_question = clean_markdown(question)
            clean_answer = clean_markdown(answer)
            
            documents.append({
                'text': f"PREGUNTA: {clean_question}\nRESPUESTA: {clean_answer}",
                'source': filename,
                'type': 'pregunta_frecuente',
                'pregunta': clean_question,
                'respuesta': clean_answer
            })
    
    # Procesar sección de migración PGI
    migracion_match = re.search(r'## MIGRACION[^#]+', content, re.IGNORECASE | re.DOTALL)
    if migracion_match:
        migracion_text = clean_markdown(migracion_match.group(0))
        documents.append({
            'text': f"MIGRACIÓN PLATAFORMA PGI:\n{migracion_text}",
            'source': filename,
            'type': 'migracion_pgi'
        })
    
    logger.info(f"❓ Preguntas frecuentes procesadas: {len(documents)} entradas")
    return documents

def process_procesos_file(content, filename):
    """Procesar archivos de procesos y mapas de tesis"""
    documents = []
    
    # ⭐ EXTRAER METADATOS YAML al inicio
    yaml_metadata = {}
    yaml_match = re.search(r'^---\n(.+?)\n---', content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1)
        for line in yaml_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                yaml_metadata[key.strip()] = value.strip()
    
    # Procesar actores del proceso (tablas)
    actores_section = re.search(r'## 👥 ACTORES DEL PROCESO(.+?)(?=##|\n---|\Z)', content, re.DOTALL | re.IGNORECASE)
    if actores_section:
        table_matches = re.findall(r'\|\s*(.+?)\s*\|\s*(.+?)\s*\|', actores_section.group(1))
        for rol, descripcion in table_matches:
            if rol.strip() and not rol.startswith('---') and not rol.startswith('Rol'):
                clean_rol = clean_markdown(rol)
                clean_desc = clean_markdown(descripcion)
                
                documents.append({
                    'text': f"ACTOR DEL PROCESO TESIS:\nROL: {clean_rol}\nDESCRIPCIÓN: {clean_desc}",
                    'source': filename,
                    'type': 'actor_proceso',
                    'rol': clean_rol,
                    'descripcion': clean_desc,
                    'keywords': yaml_metadata.get('PALABRAS_CLAVE', '')
                })
    
    # Procesar etapas del proceso
    etapas_section = re.search(r'## 📊 ETAPAS DEL PROCESO DE TESIS(.+?)(?=##|\n---|\Z)', content, re.DOTALL | re.IGNORECASE)
    if etapas_section:
        # Dividir por etapas (## 🚀 ETAPA X: ...)
        etapas = re.split(r'\n##\s+', etapas_section.group(1))
        
        for etapa in etapas:
            if not etapa.strip():
                continue
            
            # Extraer título de etapa
            titulo_match = re.match(r'(.+?)(?:\n|$)', etapa)
            if titulo_match:
                titulo = titulo_match.group(1).strip()
                contenido = etapa[len(titulo):].strip()
                
                clean_contenido = clean_markdown(contenido)
                
                documents.append({
                    'text': f"ETAPA DE TESIS: {titulo}\nCONTENIDO: {clean_contenido}",
                    'source': filename,
                    'type': 'etapa_proceso',
                    'etapa': titulo,
                    'keywords': yaml_metadata.get('PALABRAS_CLAVE', '')
                })
    
    # Si no se encontraron secciones específicas, procesar como genérico
    if not documents:
        return process_generic_file(content, filename)
    
    logger.info(f"📋 Procesos tesis procesados: {len(documents)} elementos")
    return documents

def process_reglamento_file(content, filename):
    """Procesar archivo de reglamento de tesis"""
    documents = []
    
    # Dividir por secciones principales del reglamento
    sections = re.split(r'\n##\s+', content)
    
    for section in sections:
        if not section.strip():
            continue
        
        # Extraer título de sección
        lines = section.split('\n', 1)
        title = lines[0].strip() if lines else filename
        body = lines[1] if len(lines) > 1 else section
        
        # ⭐ IDENTIFICAR TIPOS ESPECÍFICOS DE SECCIONES
        section_type = 'reglamento_general'
        
        if 'etapa' in title.lower() and any(char in title for char in '1234567890'):
            section_type = 'etapa_reglamento'
        elif 'introducción' in title.lower():
            section_type = 'introduccion_reglamento'
        elif 'carga' in title.lower():
            section_type = 'etapa_carga'
        elif 'revisión' in title.lower() or 'revision' in title.lower():
            section_type = 'etapa_revision'
        elif 'sorteo' in title.lower():
            section_type = 'etapa_sorteo'
        elif 'jurado' in title.lower():
            section_type = 'reglamento_jurados'
        
        clean_body = clean_markdown(body)
        
        # Chunking inteligente para reglamento (párrafos más largos permitidos)
        paragraphs = re.split(r'\n\n+', clean_body)
        
        current_chunk = []
        current_words = 0
        max_words = 600  # ⭐ Más palabras para texto legal
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            para_words = len(para.split())
            
            if current_words + para_words > max_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                documents.append({
                    'text': f"REGLAMENTO TESIS - {title}:\n{chunk_text}",
                    'source': filename,
                    'type': section_type,
                    'seccion': title,
                    'keywords': 'reglamento, tesis, normas, procedimientos'
                })
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            documents.append({
                'text': f"REGLAMENTO TESIS - {title}:\n{chunk_text}",
                'source': filename,
                'type': section_type,
                'seccion': title,
                'keywords': 'reglamento, tesis, normas, procedimientos'
            })
    
    logger.info(f"📚 Reglamento procesado: {len(documents)} secciones")
    return documents

def process_generic_file(content, filename):
    """Procesar archivos genéricos con chunking inteligente"""
    documents = []
    
    # Dividir por secciones (##)
    sections = re.split(r'\n##\s+', content)
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.split('\n', 1)
        title = lines[0].strip() if lines else filename
        body = lines[1] if len(lines) > 1 else section
        
        # Chunking por párrafos
        paragraphs = re.split(r'\n\n+', body)
        
        current_chunk = []
        current_words = 0
        max_words = 400
        
        for para in paragraphs:
            para_clean = clean_markdown(para.strip())
            if not para_clean:
                continue
            
            para_words = len(para_clean.split())
            
            if current_words + para_words > max_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                documents.append({
                    'text': chunk_text,
                    'source': filename,
                    'section': title,
                    'type': 'general'
                })
                current_chunk = [para_clean]
                current_words = para_words
            else:
                current_chunk.append(para_clean)
                current_words += para_words
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            documents.append({
                'text': chunk_text,
                'source': filename,
                'section': title,
                'type': 'general'
            })
    
    return documents

def create_knowledge_base(docs_folder='docs', 
                         index_file='faiss_index.bin',
                         json_file='knowledge_base.json'):
    """Crear base de conocimiento FAISS desde archivos Markdown"""
    
    logger.info("="*60)
    logger.info("CREANDO BASE DE CONOCIMIENTO UNA PUNO")
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
    
    # 3. Procesar todos los archivos con el NUEVO procesador
    all_documents = []
    for md_file in md_files:
        logger.info(f"Procesando: {md_file.name}")
        docs = parse_una_puno_markdown(md_file)  # ⭐ NUEVO PROCESADOR
        all_documents.extend(docs)
        logger.info(f"  → {len(docs)} chunks optimizados")
    
    if not all_documents:
        logger.error("No se extrajeron documentos")
        return False
    
    # ⭐ ESTADÍSTICAS MEJORADAS
    type_counts = {}
    for doc in all_documents:
        doc_type = doc.get('type', 'general')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    logger.info(f"\n📊 ESTADÍSTICAS DE CHUNKS:")
    for doc_type, count in type_counts.items():
        logger.info(f"   {doc_type}: {count} chunks")
    logger.info(f"   TOTAL: {len(all_documents)} chunks")
    
    # 4. Generar embeddings
    logger.info("Generando embeddings...")
    texts = [doc['text'] for doc in all_documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 5. Crear índice FAISS
    logger.info("Creando índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # 6. Guardar ÍNDICE FAISS
    logger.info(f"Guardando índice FAISS en {index_file}...")
    faiss.write_index(index, index_file)
    
    # 7. Guardar DOCUMENTOS con metadatos enriquecidos
    logger.info(f"Guardando documentos en {json_file}...")
    knowledge_data = {
        'documents': all_documents,
        'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'total_docs': len(all_documents),
        'dimension': dimension,
        'chunk_types': type_counts,  # ⭐ NUEVO: estadísticas
        'created_at': str(np.datetime64('now'))
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
    
    logger.info("="*60)
    logger.info("✅ BASE DE CONOCIMIENTO UNA PUNO CREADA EXITOSAMENTE")
    logger.info(f"   Archivos procesados: {len(md_files)}")
    logger.info(f"   Chunks totales: {len(all_documents)}")
    logger.info(f"   Dimensión embeddings: {dimension}")
    for doc_type, count in type_counts.items():
        logger.info(f"   {doc_type}: {count}")
    logger.info(f"   Archivo índice: {index_file}")
    logger.info(f"   Archivo JSON: {json_file}")
    logger.info("="*60)
    
    return True

if __name__ == '__main__':
    success = create_knowledge_base()
    exit(0 if success else 1)