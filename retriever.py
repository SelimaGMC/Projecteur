import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.vectorstores.chroma import Chroma
VECT_STORE = Chroma
RETRIEVER_DIR = "./chroma_db"

# ------------- A EXECUTER : ollama pull bge-m3 -------------
from langchain_community.embeddings import OllamaEmbeddings
# bge-m3 remplace nomic-embed-text : modèle multilingue avec un meilleur support
# du français (cf. ameliorations_rag.md). Dimension différente (1024 vs 768) :
# ./chroma_db doit être reconstruit après ce changement.
EMBEDDING_MODEL = "bge-m3"
OLLAMA_EMBEDDING = OllamaEmbeddings(model=EMBEDDING_MODEL)

# On n'importe plus la base SQL
from film_parser import extract_movie_sections


# Préfixe ajouté à chaque chunk indexé (Contextual Retrieval, cf. ameliorations_rag.md) :
# permet au LLM d'identifier à quel film se rapporte un extrait, même isolé du reste
# du document, et évite les confusions entre films lors de la génération.
CONTEXT_HEADER = "[Film : {titre}]\n"


def _add_context_header(chunks: list[Document], titre: str) -> list[Document]:
    """Préfixe chaque chunk par le titre du film (Contextual Retrieval)."""
    for chunk in chunks:
        chunk.page_content = CONTEXT_HEADER.format(titre=titre) + chunk.page_content
    return chunks


def format_structured_data_to_docs(sql_data: dict, url: str, titre: str) -> list[Document]:
    """Transforme les données structurées en documents textuels indexables par Chroma."""
    docs = []

    # 1. Fiche technique
    fiche = sql_data.get("fiche_technique", {})
    if fiche:
        fiche_text = "Fiche technique :\n" + "\n".join([f"- {k}: {v}" for k, v in fiche.items()])
        docs.append(Document(page_content=fiche_text, metadata={"source": url, "type": "fiche_technique", "titre": titre}))

    # 2. Distribution (Casting)
    distribution = sql_data.get("distribution", [])
    if distribution:
        distrib_text = "Distribution (Casting) :\n" + "\n".join([f"- {actor}" for actor in distribution])
        docs.append(Document(page_content=distrib_text, metadata={"source": url, "type": "distribution", "titre": titre}))

    # 3. Distinctions
    distinctions = sql_data.get("distinctions", [])
    if distinctions:
        distinc_text = "Distinctions et récompenses :\n" + "\n".join([f"- {prix}" for prix in distinctions])
        docs.append(Document(page_content=distinc_text, metadata={"source": url, "type": "distinctions", "titre": titre}))

    return docs


def build_knowledge_base(film_urls: list[str], retriever_dir=RETRIEVER_DIR, embedding=OLLAMA_EMBEDDING) -> VECT_STORE | None:
    if os.path.exists(retriever_dir) and os.listdir(retriever_dir) :
        db = Chroma(persist_directory=retriever_dir, embedding_function=embedding)
        if db._collection.count() > 0:
            print("Base vectorielle existante trouvée, chargement...")
            return db
        print("Base vectorielle existante vide, réindexation...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    for i, url in enumerate(film_urls):
        try:
            text, sql_data = extract_movie_sections(url)
            titre = sql_data.get("titre", url)

            if text:
                chunks = splitter.split_documents([Document(page_content=text, metadata={"source": url, "type": "synopsis_histoire", "titre": titre})])
                all_chunks.extend(_add_context_header(chunks, titre))

            # --- TRAITEMENT DES DONNÉES STRUCTURÉES ---
            if sql_data:
                structured_docs = format_structured_data_to_docs(sql_data, url, titre)
                # On les passe aussi au splitter au cas où une liste de casting serait exceptionnellement longue
                structured_chunks = splitter.split_documents(structured_docs)
                all_chunks.extend(_add_context_header(structured_chunks, titre))

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

    BATCH_SIZE = 50
    MAX_WORKERS = 4
    print(f"\nCalcul des embeddings et création de Chroma par lots de {BATCH_SIZE} x {MAX_WORKERS} threads...")

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

    print(f"--- Titre détecté : {sql_data['titre']} ---")

    print(f"\n--- Texte extrait ({len(text)} caractères) ---")
    print(text[:3000])
    if len(text) > 3000:
        print(f"... [tronqué, {len(text) - 3000} caractères supplémentaires]")

    print(f"\n--- Fiche technique ({len(sql_data['fiche_technique'])} entrées) ---")
    for k, v in list(sql_data['fiche_technique'].items())[:5]:
        print(f"  {k}: {v}")

    print(f"\n--- Distribution ({len(sql_data['distribution'])} entrées) ---")
    for entry in sql_data['distribution'][:5]:
        print(f"  {entry}")

    print(f"\n--- Chunks indexés (avec préfixe contextuel) ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([Document(page_content=text, metadata={"source": test_url, "titre": sql_data['titre']})])
    chunks = _add_context_header(chunks, sql_data['titre'])
    print(f"  {len(chunks)} chunks créés")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  [Chunk {i+1}] ({len(chunk.page_content)} chars)")
        print(f"  {chunk.page_content[:300]}...")