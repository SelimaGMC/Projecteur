import os
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
import secrets
import genai

os.environ["GOOGLE_API_KEY"] = my_secrets.GOOGLE_API_KEY
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])