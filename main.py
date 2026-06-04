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
import requests
from bs4 import BeautifulSoup
import wikipedia
import re

#client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
os.environ["GOOGLE_API_KEY"] = secrets.GOOGLE_API_KEY
#client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])


from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def website_to_doc(link: str):
    loader = WebBaseLoader(link)
    docs = loader.load()

    final_text = docs[0].page_content

    #Utiliser des regex et splits ici pour récupérer les sections qui nous intéressent

    # Convert the text to LangChain's `Document` format
    docs = [Document(page_content=final_text, metadata={"source": "local"})]
    vectorstore = Chroma.from_documents(
                     documents=docs,                 # Data
                     embedding=gemini_embeddings,    # Embedding model
                     persist_directory="./chroma_db" # Directory to save data
                     )
    return vectorstore


def get_movie_titles(wiki_url):
    
    wikipedia.set_lang("fr")
    response = requests.get(wiki_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    tables = soup.find_all('table', {'class': 'wikitable'})
    
    movie_titles = set() 
    
    for table in tables[:2]:
        rows = table.find_all('tr')
        for row in rows[1:]: # Ignorer les en-têtes
            cols = row.find_all(['td', 'th'])
            if len(cols) > 1:
                title_cell = cols[1]
                title = title_cell.get_text(strip=True)
                
                # Nettoyage des annotations type "[1]", "[2]" souvent présentes sur Wikipedia
                clean_title = re.sub(r'\[.*?\]', '', title).strip()
                if clean_title:
                    movie_titles.add(clean_title)
    return movie_titles

def load_movies(wiki_url: str) -> list[str]:
    
    movie_titles = get_movie_titles(wiki_url)
                    
    print(f"{len(movie_titles)} films uniques trouvés. Interrogation de l'API Wikipedia...")
    
    movie_urls = []
    for title in movie_titles:
        try:
            # auto_suggest=False évite de se retrouver sur une page inattendue si le titre est très court
            page = wikipedia.page(title, auto_suggest=False)
            movie_urls.append(page.url)
            
        except wikipedia.exceptions.DisambiguationError as e:
            # En cas d'homonymie (ex: "Lucy"), on essaie d'ajouter " (film)"
            try:
                page = wikipedia.page(f"{title} (film)")
                movie_urls.append(page.url)
            except:
                pass # Si on ne trouve toujours pas, on ignore
                
        except wikipedia.exceptions.PageError:
            # La page n'existe pas avec ce titre exact
            pass
            
    print(movie_urls)
    return movie_urls

load_movies("https://fr.wikipedia.org/wiki/Liste_des_plus_gros_succ%C3%A8s_fran%C3%A7ais_au_box-office_mondial")