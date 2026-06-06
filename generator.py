from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import VECT_STORE    # Chroma

def create_rag_chain(vectorstore: VECT_STORE):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Premier exemple de LLM (Section Generator dans le notebook)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # type: ignore[call-arg]

    prompt = ChatPromptTemplate.from_template(
        "Tu es un expert du cinéma français. Réponds à la question en te basant "
        "uniquement sur le contexte fourni. Si tu ne sais pas, dis-le.\n\n"
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