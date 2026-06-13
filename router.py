# router.py — Routeur déterministe pour l'Adaptive RAG
# Ne modifie aucun fichier existant.

from dataclasses import dataclass

@dataclass
class RetrievalStrategy:
    k: int
    search_type: str                  # "similarity" ou "mmr"
    metadata_filter: dict | None      # filtre Chroma sur les métadonnées
    description: str                  # pour le debug / les logs

_FACTUELLE_TRIGGERS = [
    "act", "réalis", "casting", "distribution", "écri", "joue", "joua", "joué"
    "durée", "année", "date", "récompense", "césar", "oscar", "budget", "distinction"
    "produ", "scénariste", "genre", "nationalité", "combien", "quand", "sorti"
]

_COMPARAISON_TRIGGERS = [
    "compare", "différence", "meilleur", "versus", "vs", "plus que",
    "moins que", "entre", "quel film a le plus", "quel est le film le plus"
]

# Les 3 stratégies de retrieval
STRATEGY_FACTUELLE = RetrievalStrategy(
    k=4,
    search_type="similarity",
    # On cible uniquement les chunks structurés — tu as déjà ces métadonnées dans retriever.py
    metadata_filter={"type": {"$in": ["fiche_technique", "distribution", "distinctions"]}},
    description="factuelle → k=4, chunks structurés uniquement"
)

STRATEGY_NARRATIVE = RetrievalStrategy(
    k=8,
    search_type="similarity",
    metadata_filter=None,             # pas de filtre, on cherche dans tout le corpus
    description="narrative → k=8, corpus complet"
)

STRATEGY_COMPARAISON = RetrievalStrategy(
    k=14,
    search_type="mmr",                # MMR pour maximiser la diversité entre films
    metadata_filter=None,
    description="comparaison → k=14, MMR pour diversité"
)


def classify_question(question: str) -> str:
    """Classifie la question en tant que factuelle, narrative ou comparaison."""
    q = question.lower()
    if any(trigger in q for trigger in _COMPARAISON_TRIGGERS):
        return "comparaison"
    if any(trigger in q for trigger in _FACTUELLE_TRIGGERS):
        return "factuelle"
    return "narrative"


def route(question: str) -> RetrievalStrategy:
    """Retourne la stratégie de retrieval adaptée au type de la question"""
    q_type = classify_question(question)
    strategy = {
        "factuelle":   STRATEGY_FACTUELLE,
        "narrative":   STRATEGY_NARRATIVE,
        "comparaison": STRATEGY_COMPARAISON,
    }[q_type]
    return strategy