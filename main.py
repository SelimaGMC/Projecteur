import os
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
import secrets
import genai

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
os.environ["GOOGLE_API_KEY"] = secrets.GOOGLE_API_KEY
#client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])


loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/")
docs = loader.load()



def charger_film_wikipedia(titre_film, nom_realisateur, langue="fr"):
    # 1. Charger la page du film
    docs_film = WikipediaLoader(query=titre_film, load_max_docs=1, lang=langue).load()
    
    # 2. Charger la page du réalisateur
    docs_realisateur = WikipediaLoader(query=nom_realisateur, load_max_docs=1, lang=langue).load()
    
    if not docs_film or not docs_realisateur:
        return []
        
    # 3. Créer un document unifié avec des métadonnées
    contenu_combine = f"FILM: {titre_film}\n\n" + docs_film[0].page_content + f"\n\nREALISATEUR: {nom_realisateur}\n\n" + docs_realisateur[0].page_content
    
    doc_final = Document(
        page_content=contenu_combine,
        metadata={
            "source": "wikipedia",
            "titre_film": titre_film,
            "realisateur": nom_realisateur,
            "type": "film_et_equipe"
        }
    )
    return [doc_final]

# Exemple d'utilisation
documents = charger_film_wikipedia("Inception (film)", "Christopher Nolan")
