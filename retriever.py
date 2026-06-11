import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.vectorstores.chroma import Chroma
VECT_STORE = Chroma
RETRIEVER_DIR = "./chroma_db"

from langchain_community.embeddings import OllamaEmbeddings
# "nomic-embed-text" est un excellent modèle d'embedding léger. 
# (à télécharger via: ollama pull nomic-embed-text)
OLLAMA_EMBEDDING = OllamaEmbeddings(model="nomic-embed-text")

from film_parser import extract_movie_sections
from database import init_db, save_film, DB_PATH

def build_knowledge_base(film_urls: list[str], retriever_dir=RETRIEVER_DIR, embedding=OLLAMA_EMBEDDING, db_path=DB_PATH) -> VECT_STORE | None:
    # il ne faut pas tout réindexer (on vérifie si Chroma existe déjà sur le disque)
    if os.path.exists(retriever_dir) and os.listdir(retriever_dir):
        db = Chroma(persist_directory=retriever_dir, embedding_function=embedding)
        if db._collection.count() > 0:
            print("Base vectorielle existante trouvée, chargement...")
            return db
        print("Base vectorielle existante vide, réindexation...")

    sql_conn = init_db(db_path)

    # le séparateur en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for i, url in enumerate(film_urls):
        try:
            text, sql_data = extract_movie_sections(url)
            save_film(sql_conn, sql_data)
            if text:
                chunks = splitter.split_documents([Document(page_content=text, metadata={"source": url})])
                all_chunks.extend(chunks)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(film_urls)} pages traitées ({len(all_chunks)} chunks)")
        except Exception as e:
            print(f"  Erreur pour {url}: {e}")

    texts = [chunk.page_content for chunk in all_chunks]
    metadatas = [chunk.metadata for chunk in all_chunks]
    ids = [f"doc_{i}" for i in range(len(texts))]

    if not texts:
        print("Erreur : Aucun texte n'a été extrait. Impossible de créer la base.")
        return None

    BATCH_SIZE = 50  # On traite les requêtes 50 par 50
    MAX_WORKERS = 4  # Nombre de lots embeddés en parallèle (cf. OLLAMA_NUM_PARALLEL côté serveur)
    print(f"\nCalcul des embeddings et création de Chroma par lots de {BATCH_SIZE}...")

    vectorstore = Chroma(persist_directory=retriever_dir, embedding_function=embedding)

    def _embed_batch(start, end):
        return start, end, embedding.embed_documents(texts[start:end])

    batches = [(start, min(start + BATCH_SIZE, len(texts))) for start in range(0, len(texts), BATCH_SIZE)]
    indexed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_embed_batch, start, end) for start, end in batches]
        for future in as_completed(futures):
            start, end, embeddings = future.result()
            vectorstore._collection.upsert(
                embeddings=embeddings,
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
            indexed += end - start
            print(f"  [Progression] {indexed}/{len(texts)} chunks indexés.")
    return vectorstore

# ================================== Tester le retriever ==================================

if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://fr.wikipedia.org/wiki/Les_Trois_Frères"

    print(f"=== Test extraction : {test_url} ===\n")
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(test_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    output = soup.find('div', {'class': 'mw-parser-output'})

    text, sql_data = extract_movie_sections(test_url)

    print(f"--- Texte extrait ({len(text)} caractères) ---")
    print(text[:3000])
    if len(text) > 3000:
        print(f"... [tronqué, {len(text) - 3000} caractères supplémentaires]")

    print(f"\n--- Fiche technique ({len(sql_data['fiche_technique'])} entrées) ---")
    for k, v in list(sql_data['fiche_technique'].items())[:5]:
        print(f"  {k}: {v}")

    print(f"\n--- Distribution ({len(sql_data['distribution'])} entrées) ---")
    for entry in sql_data['distribution'][:5]:
        print(f"  {entry}")

    print(f"\n--- Chunks indexés ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([Document(page_content=text, metadata={"source": test_url})])
    print(f"  {len(chunks)} chunks créés")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  [Chunk {i+1}] ({len(chunk.page_content)} chars)")
        print(f"  {chunk.page_content[:300]}...")