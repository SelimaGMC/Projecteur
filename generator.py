from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

from retriever import VECT_STORE    # Chroma

def create_rag_chain(vectorstore: VECT_STORE):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Tu es un expert du cinéma français. Réponds à la question en te basant "
        "uniquement sur le contexte fourni. Certaines informations sont sous forme de listes,"
        "tu pourras chercher si les mots clés correspondent à une ou plusieurs lignes en particulier et te baser sur "
        "le reste de la ligne pour répondre. Si l'information ne se trouve pas dans ces listes, cherche ailleurs parmi le contexte fourni"
        "Si tu ne sais pas, dis-le.\n\n"
        "Contexte : {context}\n"
        "Question : {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )