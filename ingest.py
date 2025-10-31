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
    """Procesador CORREGIDO para formato estándar UNA Puno"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        filename = filepath.stem
        
        # ⭐ EXTRAER METADATOS YAML - CORREGIDO
        yaml_metadata = {}
        yaml_match = re.search(r'^---\s*\n(.+?)\n---\s*\n', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            for line in yaml_content.split('\n'):
                if ':' in line and line.strip():
                    key, value = line.split(':', 1)
                    yaml_metadata[key.strip()] = value.strip()
            # Remover YAML del contenido
            content = content[yaml_match.end():].strip()
        
        file_type = yaml_metadata.get('TIPO', 'general')
        entidad = yaml_metadata.get('ENTIDAD', 'general')
        
        logger.info(f"🔍 Procesando {filename} - Tipo: {file_type}")
        
        # ⭐ ESTRATEGIA ESPECÍFICA POR TIPO DE ARCHIVO - MEJORADO
        
        # 1. ARCHIVO DE COORDINADORES
        if 'coordinador' in filename.lower():
            return process_coordinadores_file(content, filename, file_type, entidad)
        
        # 2. ARCHIVO DE LÍNEAS DE INVESTIGACIÓN
        elif any(word in filename.lower() for word in ['linea', 'sublinea', 'investigacion']):
            return process_lineas_investigacion_file(content, filename, file_type, entidad)
        
        # 3. ARCHIVO DE PREGUNTAS FRECUENTES
        elif any(word in filename.lower() for word in ['pregunta', 'faq', 'frecuente']):
            return process_preguntas_frecuentes_file(content, filename, file_type, entidad)
        
        # 4. ARCHIVO DE REGLAMENTO
        elif 'reglamento' in filename.lower():
            return process_reglamento_file(content, filename, file_type, entidad)
        
        # 5. ARCHIVO DE PROCESOS
        elif any(word in filename.lower() for word in ['proceso', 'mapa']):
            return process_procesos_file(content, filename, file_type, entidad)
        
        # ESTRATEGIA GENÉRICA MEJORADA
        else:
            return process_generic_file(content, filename, file_type, entidad)
            
    except Exception as e:
        logger.error(f"❌ Error procesando {filepath}: {e}")
        return []

def process_coordinadores_file(content, filename, file_type, entidad):
    """Procesador ESPECÍFICO para coordinadores - CORREGIDO"""
    documents = []
    
    # Buscar todas las facultades (## FACULTAD_DE_...)
    facultades = re.findall(r'##\s+(FACULTAD_DE_[^\n]+)', content)
    logger.info(f"   🏫 Facultades encontradas: {len(facultades)}")
    
    for facultad in facultades:
        # Extraer sección de la facultad
        section_pattern = rf'##\s+{re.escape(facultad)}(.+?)(?=##\s+FACULTAD_DE_|\Z)'
        section_match = re.search(section_pattern, content, re.DOTALL)
        
        if section_match:
            section_content = section_match.group(1)
            
            # Extraer campos estandarizados
            campos = extract_standard_fields(section_content)
            
            if campos:
                # Crear documento estructurado
                doc_text = f"FACULTAD: {facultad}\n"
                
                # Agregar campos importantes
                if campos.get('nombre'):
                    doc_text += f"COORDINADOR: {campos['nombre']}\n"
                if campos.get('email'):
                    doc_text += f"EMAIL: {campos['email']}\n"
                if campos.get('telefono'):
                    doc_text += f"TELÉFONO: {campos['telefono']}\n"
                if campos.get('horario'):
                    doc_text += f"HORARIO: {campos['horario']}\n"
                if campos.get('ubicacion'):
                    doc_text += f"UBICACIÓN: {campos['ubicacion']}\n"
                if campos.get('atencion'):
                    doc_text += f"ATENCIÓN: {campos['atencion']}\n"
                if campos.get('alias'):
                    doc_text += f"ALIAS: {campos['alias']}\n"
                
                # Agregar contenido completo para contexto
                doc_text += f"\nINFORMACIÓN COMPLETA:\n{clean_markdown(section_content)}"
                
                documents.append({
                    'text': doc_text,
                    'source': filename,
                    'type': 'coordinador',
                    'facultad': facultad,
                    'entidad': entidad,
                    'metadata': campos
                })
    
    logger.info(f"   📞 Coordinadores procesados: {len(documents)}")
    return documents

def process_lineas_investigacion_file(content, filename, file_type, entidad):
    """Procesador ESPECÍFICO para líneas de investigación - CORREGIDO"""
    documents = []
    
    logger.info(f"   🔬 Procesando líneas de investigación desde: {filename}")
    
    # ⭐ PATRÓN CORREGIDO - más flexible con espacios y formato
    lineas_pattern = r'##\s+(FACULTAD_DE_[^\n]+).*?###\s+(ESCUELA_DE_[^\n]+).*?\*\*TIPO:\*\*\s*linea_investigacion.*?\*\*LINEA:\*\*\s*([^\n]+).*?\*\*SUBLINEAS:\*\*\s*([^\n]+)'
    
    matches = re.findall(lineas_pattern, content, re.DOTALL | re.IGNORECASE)
    
    logger.info(f"   📊 Líneas encontradas con patrón principal: {len(matches)}")
    
    for facultad, escuela, linea, sublineas in matches:
        # Limpiar y normalizar los textos
        facultad = facultad.strip()
        escuela = escuela.strip() 
        linea = clean_markdown(linea.strip())
        sublineas = clean_markdown(sublineas.strip())
        
        doc_text = f"FACULTAD: {facultad}\nESCUELA: {escuela}\n"
        doc_text += f"LÍNEA: {linea}\nSUBÍNEAS: {sublineas}"
        
        documents.append({
            'text': doc_text,
            'source': filename,
            'type': 'linea_investigacion',
            'facultad': facultad,
            'escuela': escuela,
            'entidad': entidad,
            'metadata': {
                'linea': linea,
                'sublineas': sublineas
            }
        })
    
    # ⭐ MÉTODO ALTERNATIVO SI EL PRIMERO NO FUNCIONA
    if len(documents) == 0:
        logger.info("   ⚡ Usando método alternativo de búsqueda...")
        documents = process_lineas_alternativo(content, filename, entidad)
    
    logger.info(f"   ✅ Líneas investigación procesadas: {len(documents)}")
    return documents

def process_lineas_alternativo(content, filename, entidad):
    """Método alternativo más robusto para líneas de investigación"""
    documents = []
    
    # Buscar por facultades primero
    facultad_pattern = r'##\s+(FACULTAD_DE_[^\n]+)'
    facultades = re.findall(facultad_pattern, content)
    
    for facultad in facultades:
        # Extraer sección de cada facultad
        section_pattern = rf'##\s+{re.escape(facultad)}(.+?)(?=##\s+FACULTAD_DE_|\Z)'
        section_match = re.search(section_pattern, content, re.DOTALL)
        
        if section_match:
            facultad_content = section_match.group(1)
            
            # Buscar escuelas dentro de esta facultad
            escuela_pattern = r'###\s+(ESCUELA_DE_[^\n]+)'
            escuelas = re.findall(escuela_pattern, facultad_content)
            
            for escuela in escuelas:
                # Extraer sección de cada escuela
                escuela_section_pattern = rf'###\s+{re.escape(escuela)}(.+?)(?=###\s+ESCUELA_DE_|\Z)'
                escuela_match = re.search(escuela_section_pattern, facultad_content, re.DOTALL)
                
                if escuela_match:
                    escuela_content = escuela_match.group(1)
                    
                    # ⭐ BUSCAR CAMPOS CON PATRONES MÁS FLEXIBLES
                    campos = {}
                    
                    # Buscar línea
                    linea_match = re.search(r'\*\*LINEA:\*\*\s*([^\n]+)', escuela_content, re.IGNORECASE)
                    if linea_match:
                        campos['linea'] = clean_markdown(linea_match.group(1).strip())
                    
                    # Buscar sublíneas  
                    sublineas_match = re.search(r'\*\*SUBLINEAS:\*\*\s*([^\n]+)', escuela_content, re.IGNORECASE)
                    if sublineas_match:
                        campos['sublineas'] = clean_markdown(sublineas_match.group(1).strip())
                    
                    # Solo crear documento si tenemos ambos campos
                    if campos.get('linea') and campos.get('sublineas'):
                        doc_text = f"FACULTAD: {facultad}\nESCUELA: {escuela}\n"
                        doc_text += f"LÍNEA: {campos['linea']}\nSUBÍNEAS: {campos['sublineas']}"
                        
                        documents.append({
                            'text': doc_text,
                            'source': filename,
                            'type': 'linea_investigacion',
                            'facultad': facultad,
                            'escuela': escuela,
                            'entidad': entidad,
                            'metadata': campos
                        })
    
    return documents

def process_preguntas_frecuentes_file(content, filename, file_type, entidad):
    """Procesador para preguntas frecuentes - CORREGIDO"""
    documents = []
    
    # Buscar entradas de preguntas frecuentes
    # Patrón: ## TITULO\n**TIPO:** pregunta_frecuente\n**PREGUNTA:** ...\n**RESPUESTA:** ...
    preguntas_pattern = r'##\s+([^\n]+)\s*\*\*TIPO:\*\* pregunta_frecuente.*?\*\*PREGUNTA:\*\*([^\n]+).*?\*\*RESPUESTA:\*\*([^\n]+)'
    matches = re.findall(preguntas_pattern, content, re.DOTALL)
    
    for titulo, pregunta, respuesta in matches:
        doc_text = f"PREGUNTA: {pregunta.strip()}\nRESPUESTA: {respuesta.strip()}"
        
        documents.append({
            'text': doc_text,
            'source': filename,
            'type': 'pregunta_frecuente',
            'pregunta': pregunta.strip(),
            'respuesta': respuesta.strip(),
            'entidad': entidad
        })
    
    # Si no se encontraron con el patrón anterior, buscar por líneas que contengan **TIPO:** pregunta_frecuente
    if len(documents) == 0:
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            if '**TIPO:** pregunta_frecuente' in lines[i]:
                # Buscar pregunta y respuesta en líneas adyacentes
                pregunta = ""
                respuesta = ""
                j = i
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('##'):
                    if '**PREGUNTA:**' in lines[j]:
                        pregunta = lines[j].replace('**PREGUNTA:**', '').strip()
                    if '**RESPUESTA:**' in lines[j]:
                        respuesta = lines[j].replace('**RESPUESTA:**', '').strip()
                    j += 1
                
                if pregunta and respuesta:
                    doc_text = f"PREGUNTA: {pregunta}\nRESPUESTA: {respuesta}"
                    documents.append({
                        'text': doc_text,
                        'source': filename,
                        'type': 'pregunta_frecuente',
                        'pregunta': pregunta,
                        'respuesta': respuesta,
                        'entidad': entidad
                    })
                
                i = j
            else:
                i += 1
    
    logger.info(f"   ❓ Preguntas frecuentes procesadas: {len(documents)}")
    return documents

def process_reglamento_file(content, filename, file_type, entidad):
    """Procesador para reglamento - CORREGIDO"""
    documents = []
    
    # Buscar artículos del reglamento
    articulos_pattern = r'##\s+([^\n]+)\s*\*\*TIPO:\*\* articulo_reglamento.*?\*\*CONTENIDO:\*\*([^\n]+)'
    matches = re.findall(articulos_pattern, content, re.DOTALL)
    
    for titulo, contenido in matches:
        doc_text = f"ARTÍCULO: {titulo.strip()}\nCONTENIDO: {contenido.strip()}"
        
        documents.append({
            'text': doc_text,
            'source': filename,
            'type': 'articulo_reglamento',
            'articulo': titulo.strip(),
            'entidad': entidad
        })
    
    logger.info(f"   📖 Artículos de reglamento procesados: {len(documents)}")
    return documents

def process_procesos_file(content, filename, file_type, entidad):
    """Procesador para procesos - CORREGIDO"""
    documents = []
    
    # Buscar diferentes tipos de elementos de procesos
    patterns = [
        (r'##\s+([^\n]+)\s*\*\*TIPO:\*\* etapa_proceso.*?\*\*DESCRIPCION:\*\*([^\n]+)', 'etapa_proceso'),
        (r'##\s+([^\n]+)\s*\*\*TIPO:\*\* actor_proceso.*?\*\*DESCRIPCION:\*\*([^\n]+)', 'actor_proceso'),
    ]
    
    for pattern, doc_type in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for titulo, descripcion in matches:
            doc_text = f"{doc_type.upper()}: {titulo.strip()}\nDESCRIPCIÓN: {descripcion.strip()}"
            
            documents.append({
                'text': doc_text,
                'source': filename,
                'type': doc_type,
                'titulo': titulo.strip(),
                'entidad': entidad
            })
    
    logger.info(f"   📋 Elementos de proceso procesados: {len(documents)}")
    return documents

def process_generic_file(content, filename, file_type, entidad):
    """Procesador genérico mejorado"""
    documents = []
    
    # Buscar secciones ## y extraer contenido
    sections = re.split(r'\n##\s+', content)
    
    for section in sections:
        if not section.strip():
            continue
        
        # Extraer título y cuerpo
        lines = section.split('\n', 1)
        title = lines[0].strip() if lines else filename
        body = lines[1] if len(lines) > 1 else ""
        
        # Extraer campos estandarizados si existen
        campos = extract_standard_fields(body)
        
        if campos:
            # Construir texto estructurado
            doc_text = f"TÍTULO: {title}\n"
            for key, value in campos.items():
                doc_text += f"{key.upper()}: {value}\n"
        else:
            # Usar contenido limpio
            doc_text = clean_markdown(f"{title}\n{body}")
        
        if doc_text.strip():
            documents.append({
                'text': doc_text,
                'source': filename,
                'type': file_type,
                'entidad': entidad,
                'section': title
            })
    
    return documents

def extract_standard_fields(body):
    """Extraer campos estandarizados **CAMPO:** valor - CORREGIDO"""
    fields = {}
    
    if not body:
        return fields
    
    # Patrón mejorado para campos estandarizados
    field_pattern = r'\*\*([A-Z_]+):\*\*\s*([^\n]+)'
    matches = re.findall(field_pattern, body)
    
    for field_name, field_value in matches:
        clean_value = clean_markdown(field_value.strip())
        fields[field_name.lower()] = clean_value
    
    # Si no hay campos estandarizados, buscar contenido entre **
    if not fields:
        bold_pattern = r'\*\*([^\*]+)\*\*'
        bold_matches = re.findall(bold_pattern, body)
        if bold_matches:
            fields['contenido'] = ' '.join(bold_matches)
    
    return fields

def create_knowledge_base(docs_folder='docs', 
                         index_file='faiss_index.bin',
                         json_file='knowledge_base.json'):
    """Crear base de conocimiento FAISS - CORREGIDO"""
    
    logger.info("="*60)
    logger.info("CREANDO BASE DE CONOCIMIENTO UNA PUNO - CORREGIDO")
    logger.info("="*60)
    
    # 1. Cargar modelo de embeddings
    logger.info("Cargando modelo de embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Leer todos los archivos .md
    docs_path = Path(docs_folder)
    if not docs_path.exists():
        logger.error(f"❌ Carpeta {docs_folder} no existe")
        return False
    
    md_files = list(docs_path.glob('*.md'))
    if not md_files:
        logger.error(f"❌ No se encontraron archivos .md en {docs_folder}")
        return False
    
    logger.info(f"Encontrados {len(md_files)} archivos .md")
    
    # 3. Procesar todos los archivos con el PROCESADOR CORREGIDO
    all_documents = []
    for md_file in md_files:
        logger.info(f"Procesando: {md_file.name}")
        docs = parse_standard_markdown(md_file)
        all_documents.extend(docs)
        logger.info(f"  → {len(docs)} documentos creados")
    
    if not all_documents:
        logger.error("❌ No se extrajeron documentos")
        return False
    
    # Estadísticas detalladas
    type_counts = {}
    faculty_counts = {}
    
    for doc in all_documents:
        doc_type = doc.get('type', 'general')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        facultad = doc.get('facultad', '')
        if facultad:
            faculty_counts[facultad] = faculty_counts.get(facultad, 0) + 1
    
    logger.info(f"\n📊 ESTADÍSTICAS FINALES:")
    logger.info(f"   Documentos totales: {len(all_documents)}")
    logger.info(f"   Por tipo:")
    for doc_type, count in sorted(type_counts.items()):
        logger.info(f"     {doc_type}: {count}")
    
    if faculty_counts:
        logger.info(f"   Por facultad (top 5):")
        for facultad, count in sorted(faculty_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"     {facultad}: {count}")
    
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
        'faculty_counts': faculty_counts,
        'created_at': str(np.datetime64('now'))
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
    
    logger.info("="*60)
    logger.info("✅ BASE DE CONOCIMIENTO CREADA EXITOSAMENTE")
    logger.info(f"   Archivos procesados: {len(md_files)}")
    logger.info(f"   Documentos totales: {len(all_documents)}")
    logger.info(f"   Dimensión embeddings: {dimension}")
    logger.info("="*60)
    
    return True

if __name__ == '__main__':
    success = create_knowledge_base()
    exit(0 if success else 1)