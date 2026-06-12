"""Harnais d'évaluation Q/R de référence pour le RAG cinéma français.

Usage :
    python eval_qa.py <label>

Écrit eval_results_<label>.json et affiche un résumé : pour chaque question,
un score de "retrieval" (les mots-clés attendus sont-ils présents dans les
extraits récupérés ?) et un score de "correctness" (sont-ils présents dans la
réponse finale du LLM ?), agrégés par type de question et globalement.

Cela permet de comparer un "avant" et un "après" modification du pipeline,
par exemple :
    python eval_qa.py avant   # avant les changements (cf. ameliorations_rag.md)
    ...                        # appliquer les changements + réindexer
    python eval_qa.py apres

Les questions et mots-clés attendus ci-dessous sont un point de départ basé
sur des informations publiques bien connues des films listés (qui devraient
faire partie du corpus des plus gros succès français au box-office mondial).
Si le score de retrieval reste bas pour une question alors que le film est
bien indexé, ajustez la question ou les mots-clés pour qu'ils correspondent
mieux à la formulation effective des pages Wikipédia indexées.
"""

import json
import os
import sys
import unicodedata

from retriever import build_knowledge_base, RETRIEVER_DIR
from generator import create_rag_chain, build_retriever


QA_SET = [
    # --- Questions factuelles : réponse attendue dans fiche_technique / distribution / distinctions ---
    {
        "type": "factuelle",
        "question": "Qui sont les réalisateurs du film Intouchables ?",
        "expected_keywords": ["Nakache", "Toledano"],
    },
    {
        "type": "factuelle",
        "question": "Quel acteur interprète le rôle de Driss dans le film Intouchables ?",
        "expected_keywords": ["Omar Sy"],
    },
    {
        "type": "factuelle",
        "question": "Qui a réalisé le film Bienvenue chez les Ch'tis ?",
        "expected_keywords": ["Dany Boon"],
    },
    {
        "type": "factuelle",
        "question": "Quels sont les deux acteurs principaux du film La Grande Vadrouille ?",
        "expected_keywords": ["Bourvil", "Funès"],
    },
    {
        "type": "factuelle",
        "question": "Quelle actrice interprète Amélie Poulain dans Le Fabuleux Destin d'Amélie Poulain ?",
        "expected_keywords": ["Audrey Tautou"],
    },
    # --- Questions narratives : réponse attendue dans synopsis_histoire / accueil ---
    {
        "type": "narrative",
        "question": "Quel est le sujet principal du film Intouchables ?",
        "expected_keywords": ["tétraplégique", "banlieue"],
    },
    {
        "type": "narrative",
        "question": "Dans quelle région de France se déroule le film Bienvenue chez les Ch'tis ?",
        "expected_keywords": ["Nord"],
    },
    {
        "type": "narrative",
        "question": "Quel est le métier du personnage principal du film Bienvenue chez les Ch'tis ?",
        "expected_keywords": ["facteur"],
    },
    {
        "type": "narrative",
        "question": "Dans quel pays se déroule l'histoire du film Astérix et Obélix : Mission Cléopâtre ?",
        "expected_keywords": ["Égypte"],
    },
    {
        "type": "narrative",
        "question": "Quels sont les deux personnages principaux du film Les Visiteurs ?",
        "expected_keywords": ["Godefroy", "Jacquouille"],
    },
]


def normalize(text: str) -> str:
    """Minuscules + suppression des accents (matching robuste sur les noms propres français)."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def keyword_score(text: str, keywords: list[str]) -> float:
    """Fraction des mots-clés attendus présents dans le texte (0.0 à 1.0)."""
    if not keywords:
        return 1.0
    norm_text = normalize(text)
    hits = sum(1 for kw in keywords if normalize(kw) in norm_text)
    return hits / len(keywords)


def _average(results: list[dict], key: str) -> float:
    return sum(r[key] for r in results) / len(results)


def run_eval(label: str) -> None:
    if not os.path.exists(RETRIEVER_DIR) or not os.listdir(RETRIEVER_DIR):
        print(f"Erreur : '{RETRIEVER_DIR}' est vide ou absent.")
        print("Lancez d'abord `python main.py` (ou `streamlit run app.py`) pour construire la base vectorielle.")
        return

    vectorstore = build_knowledge_base([])
    if vectorstore is None:
        print("Erreur : impossible de charger la base vectorielle.")
        return

    retriever = build_retriever(vectorstore)
    chain = create_rag_chain(vectorstore)

    results = []
    for item in QA_SET:
        question = item["question"]
        keywords = item["expected_keywords"]

        docs = retriever.invoke(question)
        context = "\n".join(doc.page_content for doc in docs)
        retrieval_score = keyword_score(context, keywords)

        answer = chain.invoke(question)
        correctness_score = keyword_score(answer, keywords)

        results.append({
            "type": item["type"],
            "question": question,
            "expected_keywords": keywords,
            "retrieval_score": retrieval_score,
            "correctness_score": correctness_score,
            "answer": answer,
        })

        print(f"[{item['type']:9s}] {question}")
        print(f"             retrieval={retrieval_score:.2f}  correctness={correctness_score:.2f}")
        print(f"             -> {answer[:200]}")

    summary = {}
    for q_type in ("factuelle", "narrative"):
        subset = [r for r in results if r["type"] == q_type]
        if subset:
            summary[q_type] = {
                "n": len(subset),
                "retrieval_score": _average(subset, "retrieval_score"),
                "correctness_score": _average(subset, "correctness_score"),
            }
    summary["global"] = {
        "n": len(results),
        "retrieval_score": _average(results, "retrieval_score"),
        "correctness_score": _average(results, "correctness_score"),
    }

    output = {"label": label, "results": results, "summary": summary}
    out_path = f"eval_results_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Résumé ===")
    for q_type, stats in summary.items():
        print(f"  {q_type:10s} (n={stats['n']:2d}) : retrieval={stats['retrieval_score']:.2f}  correctness={stats['correctness_score']:.2f}")
    print(f"\nRésultats enregistrés dans {out_path}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    run_eval(label)
