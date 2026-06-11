from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
import sqlite3

from retriever import VECT_STORE    # Chroma
from database import DB_PATH        

def create_rag_chain(vectorstore: VECT_STORE):
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance pour diversité
        search_kwargs={
            "k": 20,           # Retourner 20 chunks
            "fetch_k": 100,    # Considérer 100 chunks avant sélection
            "lambda_mult": 0.5 # Équilibre entre similarité et diversité
        }
    )

    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Tu es un expert du cinéma français.\n\n"
        "Voici des extraits d'articles Wikipedia sur des films français.\n"
        "Réponds à la question en utilisant UNIQUEMENT ces informations.\n\n"
        "Si tu ne sais pas, dis-le.\n\n"
        "=== EXTRAITS ===\n"
        "{context}\n\n"
        "=== QUESTION ===\n"
        "{question}\n\n"
        "=== RÉPONSE ===\n"
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def format_docs(docs):
    """Formate les documents retournés par le retriever pour que le LLM soit capable de lire les documents"""
    if not docs:
        return "Aucun document trouvé."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Source inconnue')
        formatted.append(f"[Document {i} - {source}]\n{doc.page_content}\n")
    
    return "\n".join(formatted)