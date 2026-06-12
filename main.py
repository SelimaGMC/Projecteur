import os
import secrets

os.environ["GOOGLE_API_KEY"] = secrets.GOOGLE_API_KEY
agent = secrets.CUSTOM_AGENT

from crawler import load_movies_urls
from retriever import build_knowledge_base, RETRIEVER_DIR as _RETRIEVER_DIR, OLLAMA_EMBEDDING
from generator import create_rag_chain
from langchain_community.vectorstores.chroma import Chroma
WIKI_BOX_OFFICE_URL = "https://fr.wikipedia.org/wiki/Liste_des_plus_gros_succ%C3%A8s_fran%C3%A7ais_au_box-office_mondial"
RETRIEVER_DIR = "./chroma_db"


def _existing_collection_has_data() -> bool:
    if not os.path.exists(_RETRIEVER_DIR) or not os.listdir(_RETRIEVER_DIR):
        return False
    db = Chroma(persist_directory=_RETRIEVER_DIR, embedding_function=OLLAMA_EMBEDDING)
    return db._collection.count() > 0


def main():
    urls = []
    if not _existing_collection_has_data():
        urls, movies = load_movies_urls(WIKI_BOX_OFFICE_URL, agent=agent)
        print(f"Indexation de {len(urls)} pages Wikipedia...")
        print(f"Films disponibles: {movies}")

    vectorstore = build_knowledge_base(urls)
    chain = create_rag_chain(vectorstore)

    print("\nSystème RAG prêt. Posez vos questions sur le cinéma français (Ctrl+C pour quitter).\n")
    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue
            response = chain.invoke(question)
            print(f"\nRéponse : {response}\n")
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break


if __name__ == "__main__":
    main()