# verification.py
import json
import faiss
from sentence_transformers import SentenceTransformer

def verify_knowledge_base():
    print("🔍 VERIFICANDO BASE DE CONOCIMIENTO...")
    
    try:
        # 1. Verificar archivo JSON
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        print(f"✅ JSON cargado: {len(documents)} documentos")
        
        # Mostrar algunos documentos de ejemplo
        print("\n📄 MUESTRA DE DOCUMENTOS:")
        for i, doc in enumerate(documents[:5]):
            print(f"{i+1}. Tipo: {doc.get('type', '?')} | Facultad: {doc.get('facultad', '?')}")
            print(f"   Texto: {doc.get('text', '')[:80]}...")
            print()
        
        # 2. Verificar FAISS
        index = faiss.read_index('faiss_index.bin')
        print(f"✅ FAISS cargado: {index.ntotal} vectores")
        
        # 3. Buscar específicamente enfermería
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        query = "enfermería líneas investigación"
        query_vector = model.encode([query])
        
        distances, indices = index.search(query_vector, 10)
        
        print(f"\n🔎 BÚSQUEDA MANUAL: '{query}'")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(documents):
                doc = documents[idx]
                if 'enfermer' in doc.get('text', '').lower() or 'enfermer' in doc.get('facultad', '').lower():
                    print(f"🎯 ENCONTRADO: Sim: {1/(1+dist):.3f} | Tipo: {doc.get('type')} | Fac: {doc.get('facultad')}")
                    print(f"   Texto: {doc.get('text', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    verify_knowledge_base()