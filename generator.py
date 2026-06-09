from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
import sqlite3

from retriever import VECT_STORE    # Chroma
from database import DB_PATH        

def create_rag_chain(vectorstore: VECT_STORE):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(model="llama3.2", temperature=0)

    def custom_context_retriever(question: str) -> str:
        # 1. On cherche d'abord les morceaux de textes pertinents dans Chroma
        docs = retriever.invoke(question)
        
        # 2. On extrait les URLs uniques des films trouvés par Chroma
        urls = list(set(doc.metadata.get("source") for doc in docs if "source" in doc.metadata))
        
        # 3. On va chercher la fiche technique correspondante dans SQL
        sql_contexts = []
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for url in urls:
            try:
                query = """
                    SELECT f.url,
                        (SELECT GROUP_CONCAT(cle || ': ' || valeur, ' | ') FROM fiche_technique WHERE film_id = f.id) as fiche,
                        (SELECT GROUP_CONCAT(entree, ', ') FROM distribution WHERE film_id = f.id) as acteurs,
                        (SELECT GROUP_CONCAT(entree, ' / ') FROM distinctions WHERE film_id = f.id) as recs
                    FROM films f WHERE f.url = ?;
                """
                cursor.execute(query, (url,))
                row = cursor.fetchone()
                if row:
                    sql_contexts.append(
                        f"=== DONNÉES STRUCTURÉES SQL ({row[0]}) ===\n"
                        f"Fiche: {row[1]}\nDistribution: {row[2]}\nDistinctions: {row[3]}"
                    )
                conn.close()

            except Exception as e:
                # Sécurité pour ne pas faire planter le RAG si le nom de table diffère
                sql_contexts.append(f"[Note : Impossible de lire SQL pour {url} -> {e}]")
                
        conn.close()
        
        # On assemble les deux en un seul super-contexte
        chroma_context = "\n\n".join(doc.page_content for doc in docs)
        sql_context = "\n\n".join(sql_contexts)
        
        complete_context = (
            f"--- DONNÉES STRUCTURÉES ---\n{sql_context}\n\n"
            f"--- TEXTES ET ANECDOTES ---\n{chroma_context}"
        )
        return complete_context

    prompt = ChatPromptTemplate.from_template(
        "Tu es un expert du cinéma français. Réponds à la question en te basant "
        "uniquement sur le contexte fourni. Si tu ne sais pas, dis-le.\n\n"
        "Contexte : {context}\n"
        "Question : {question}"
    )

    return (
        {"context": RunnableLambda(custom_context_retriever), 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )