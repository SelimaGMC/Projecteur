from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase # <-- NOUVEL IMPORT
import sqlite3
import re

from retriever import VECT_STORE
from database import DB_PATH

# 1. Instanciation de l'outil de base de données LangChain.
# Cela permet à LangChain d'aller "lire" lui-même l'architecture de tes tables.
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_dynamic_schema(_) -> str:
    """
    Fonction qui s'exécutera à chaque requête.
    Elle récupère dynamiquement les 'CREATE TABLE' de ta BDD,
    ainsi que 3 lignes d'exemples par table (Crucial pour l'IA).
    """
    return db.get_table_info()

def clean_sql_query(raw_query: str) -> str:
    query = raw_query.strip()
    query = re.sub(r"^```sql\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^```json\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^```\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"```$", "", query)
    return query.strip()

def create_advanced_rag_chain(vectorstore: VECT_STORE, db_path=DB_PATH):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm_text = ChatOllama(model="llama3.2", temperature=0)
    llm_json = ChatOllama(model="llama3.2", temperature=0, format="json")

    # --- PROMPT 1 : LE ROUTEUR (inchangé) ---
    router_prompt = ChatPromptTemplate.from_template(
        "Tu es un assistant de routage expert pour une base de données sur le cinéma.\n"
        "La base structurée (SQL) contient les fiches techniques : Réalisation, Genre, Scénario, Musique, Décors, acteurs, personnages, récompenses.\n"
        "La base non-structurée (Chroma) contient les textes : synopsis complets, critiques, anecdotes de tournage, histoire.\n\n"
        "Analyse la question de l'utilisateur.\n"
        "Réponds UNIQUEMENT avec un objet JSON valide contenant exactement une clé 'source' "
        "dont la valeur doit être 'sql', 'chroma', ou 'both' (si la question nécessite les deux informations).\n\n"
        "Question : {question}"
    )
    router_chain = router_prompt | llm_json | JsonOutputParser()

    sql_generation_prompt = ChatPromptTemplate.from_template(
        "Tu es un générateur de requêtes SQLite expert.\n"
        "Voici le schéma actuel de la base de données, généré dynamiquement (inclus les tables et quelques lignes d'exemples) :\n"
        "========================\n"
        "{schema}\n"
        "========================\n\n"
        "IMPORTANT : La base utilise une structure Clé/Valeur (EAV). "
        "Voici les modèles obligatoires que tu DOIS imiter pour l'interroger :\n"
        "Exemple 1 (Chercher une info précise comme le réalisateur ou l'année) :\n"
        "Q: Qui est le réalisateur de Le Petit Prince ?\n"
        "A: SELECT f.url, ft.cle, ft.valeur FROM fiche_technique ft JOIN films f ON ft.film_id = f.id WHERE ft.cle = 'Réalisation' AND ft.film_id IN (SELECT film_id FROM fiche_technique WHERE cle = 'Titre' AND valeur LIKE '%Le Petit Prince%');\n\n"
        "Exemple 2 (Distribution) :\n"
        "Q: Qui joue dans Lucy ?\n"
        "A: SELECT f.url, 'Acteur' as cle, d.entree as valeur FROM distribution d JOIN films f ON d.film_id = f.id WHERE d.film_id IN (SELECT film_id FROM fiche_technique WHERE cle = 'Titre' AND valeur LIKE '%Lucy%');\n\n"
        "Génère UNIQUEMENT la requête SQL, sans commentaires et sans balises markdown.\n"
        "Question : {question}"
    )
    
    # La chaîne SQL intègre maintenant un "RunnablePassthrough.assign" 
    # qui va calculer "{schema}" à la volée avant de l'envoyer au LLM.
    sql_chain = (
        RunnablePassthrough.assign(schema=get_dynamic_schema) 
        | sql_generation_prompt 
        | llm_text 
        | StrOutputParser() 
        | RunnableLambda(clean_sql_query)
    )

    # --- L'ORCHESTRATEUR HYBRIDE ---
    def hybrid_orchestrator(question: str) -> str:
        try:
            route_decision = router_chain.invoke({"question": question})
            source = route_decision.get("source", "both").lower()
        except Exception as e:
            source = "both"
            
        print(f"\n[RAG Routeur] Décision prise : {source.upper()}")
        context_parts = []
        
        # B. Exécution SQL
        if source in ["sql", "both"]:
            sql_query = sql_chain.invoke({"question": question}) # La question passe ici, et "schema" est assigné automatiquement !
            print(f"[RAG SQL] Requête générée : {sql_query}")
            
            if "SELECT" in sql_query.upper():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    sql_results = cursor.fetchall()
                    
                    if sql_results:
                        col_names = [desc[0] for desc in cursor.description]
                        unique_results = list(set(sql_results))[:5] 
                        
                        formatted_rows = []
                        for row in unique_results:
                            row_dict = {col: val for col, val in zip(col_names, row)}
                            formatted_rows.append(f"- {row_dict}")
                            
                        context_parts.append(
                            f"--- DONNÉES STRUCTURÉES (Extraites en réponse à la question '{question}') ---\n" + 
                            "\n".join(formatted_rows)
                        )
                    conn.close()
                except Exception as e:
                    print(f"[RAG SQL] Erreur d'exécution : {e}")

        # C. Exécution Chroma
        if source in ["chroma", "both"] or (source == "sql" and not context_parts):
            docs = retriever.invoke(question)
            if docs:
                formatted_chroma = "\n\n".join(doc.page_content for doc in docs)
                context_parts.append(f"--- TEXTES NON-STRUCTURÉS (CHROMA) ---\n{formatted_chroma}")

        if not context_parts:
            return "AUCUN_CONTEXTE_DISPONIBLE"
            
        return "\n\n".join(context_parts)

    # --- PROMPT 3 : SYNTHÈSE FINALE ---
    final_prompt = ChatPromptTemplate.from_template(
        "Tu es un expert du cinéma français.\n"
        "Réponds à la question posée en te basant exclusivement sur les contextes fournis ci-dessous.\n"
        "Si le contexte contient 'AUCUN_CONTEXTE_DISPONIBLE', réponds : 'Je ne sais pas, l'information n'est pas stockée.'\n\n"
        "Contexte :\n{context}\n\n"
        "Question : {question}"
    )

    return (
        {
            "context": RunnableLambda(hybrid_orchestrator),
            "question": RunnablePassthrough() 
        }
        | final_prompt
        | llm_text
        | StrOutputParser()
    )