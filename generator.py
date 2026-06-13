from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda

from retriever import VECT_STORE    # Chroma

from router import route

# Modèle de génération : remplace llama3.2 (cf. ameliorations_rag.md).
# Alternative testée : "llama3.1:8b" -- comparer les deux avec eval_qa.py.
GENERATION_MODEL = "qwen2.5:7b"

# Nombre de chunks renvoyés au LLM. Remplace le retrieval MMR (k=20, fetch_k=100,
# lambda_mult=0.5) par une recherche par similarité plus ciblée : une fois chaque
# chunk préfixé par "[Film : <titre>]" (cf. retriever.py), la diversité imposée par
# MMR tend surtout à écarter des chunks pertinents du film recherché au profit
# d'autres films (cf. ameliorations_rag.md).
RETRIEVAL_K = 8

# Fenêtre de contexte explicite : k=8 chunks préfixés (~8 400 caractères, soit
# environ 2000-2500 tokens) + prompt système restent confortablement sous 8192
# tokens, ce qui évite toute troncature silencieuse côté Ollama (cf. ameliorations_rag.md).
NUM_CTX = 8192

SYSTEM_PROMPT = (
    "Tu es un assistant expert du cinéma français.\n\n"
    "Tu reçois des extraits issus d'une base de connaissances sur des films "
    "français. Chaque extrait commence par une étiquette \"[Film : <titre>]\" "
    "qui indique le film concerné, suivie d'un des types de contenu suivants :\n"
    "- un passage narratif (synopsis, accueil critique, analyse) ;\n"
    "- une \"Fiche technique\" (réalisation, genre, durée, etc.) ;\n"
    "- une liste \"Distribution (Casting)\" (acteurs et rôles) ;\n"
    "- une liste \"Distinctions et récompenses\".\n\n"
    "Règles strictes :\n"
    "1. Réponds UNIQUEMENT à partir des extraits ci-dessous. N'utilise jamais "
    "de connaissances générales pour compléter une réponse.\n"
    "2. Si l'information demandée n'apparaît dans aucun extrait, réponds "
    "exactement : \"Je ne dispose pas de cette information dans mes sources.\"\n"
    "3. Ne combine jamais des informations provenant d'extraits "
    "\"[Film : X]\" et \"[Film : Y]\" différents dans une même affirmation, "
    "sauf si la question demande explicitement une comparaison.\n"
    "4. Si plusieurs films apparaissent dans les extraits, précise le titre du "
    "film concerné par chaque information donnée.\n"
    "5. Réponds en français, de manière concise et factuelle.\n\n"
    "=== EXTRAITS ===\n"
    "{context}"
)


def build_retriever(vectorstore: VECT_STORE, question: str = ""):
    """Construit le retriever utilisé pour la génération (cf. eval_qa.py pour son
    évaluation indépendante de la génération)."""
    if not question:
        # Fallback sur la stratégie narrative par défaut
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        )

    strategy = route(question)
    search_kwargs = {"k": strategy.k}
    if strategy.metadata_filter:
        search_kwargs["filter"] = strategy.metadata_filter

    return vectorstore.as_retriever(
        search_type=strategy.search_type,
        search_kwargs=search_kwargs,
    )


def create_rag_chain(vectorstore: VECT_STORE):
    llm = ChatOllama(model=GENERATION_MODEL, temperature=0, num_ctx=NUM_CTX)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def retrieve_with_routing(inputs):
        # inputs peut être une string ou un dict selon comment chain.invoke() est appelé
        question = inputs if isinstance(inputs, str) else inputs["question"]
        retriever = build_retriever(vectorstore, question)
        docs = retriever.invoke(question)
        return {"context": format_docs(docs), "question": question}

    return RunnableLambda(retrieve_with_routing) | prompt | llm | StrOutputParser()

def format_docs(docs):
    """Concatène les chunks renvoyés par le retriever.

    Chaque chunk est déjà préfixé par "[Film : <titre>]" (cf. retriever.py),
    ce qui le rend auto-descriptif sans avoir besoin de répéter l'URL source.
    """
    if not docs:
        return "Aucun extrait pertinent trouvé."

    return "\n\n---\n\n".join(doc.page_content for doc in docs)
