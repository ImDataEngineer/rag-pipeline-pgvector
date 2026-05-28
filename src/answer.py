"""Lumora Stock — answer generator + hallucination gate (scaffold).

Ce module N'EST PAS jugé sur la qualité de la réponse LLM. Il est jugé sur
sa capacité à REFUSER DE RÉPONDRE quand le retrieval est trop faible.

CONTRAT
- `generate_answer(question, conn, model=None)` retourne :
    * `None` (ou lève `LowConfidenceRetrievalError`) si le top-1 score est
      sous un seuil — c'est le hallucination gate.
    * Un `str` non-vide sinon. Le contenu de la réponse n'est PAS vérifié.

- Le seuil par défaut est `0.35` (similarité cosine). Tu peux le déplacer
  mais documente ton choix dans le README. Les questions in-domain doivent
  passer le gate ; les 10 questions out-of-domain (foot, cuisine, météo) NE
  DOIVENT PAS le passer.

POURQUOI ÇA COMPTE
En production, un RAG qui hallucine sur une question hors-domaine est pire
qu'un RAG qui refuse poliment. Le no-answer gate est le contrôle de qualité
le plus rentable d'un RAG — il coûte 5 lignes de code, et il évite 80% des
incidents prod (« le bot a inventé un numéro de série »).
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import Connection

from src.retrieve import top_k_search

CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_MIN_SCORE", "0.35"))


class LowConfidenceRetrievalError(Exception):
    """Levée quand le top-1 score est sous le seuil de confiance.

    Tu peux RAISER cette exception OU retourner None — les deux sont acceptés
    par la rubric. Lever est plus propre côté API (le caller peut catcher
    et répondre 422), retourner None est plus simple côté script.
    """


def generate_answer(
    question: str,
    conn: Connection,
    model=None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Optional[str]:
    """Pipeline RAG complet : retrieve → gate → (optionally) generate.

    Args:
        question: la question utilisateur.
        conn: connexion SQLAlchemy active.
        model: SentenceTransformer optionnel pour embed la question.
        threshold: score minimum requis pour considérer le retrieval fiable.

    Returns:
        - `None` si le retrieval est insuffisant (hallucination gate activé).
        - Une chaîne non-vide sinon. Le contenu réel de la réponse n'est PAS
          jugé par la rubric — tu peux :
          * te contenter d'un template `"D'après les articles X, Y, Z : ..."`,
          * OU brancher Ollama (`qwen2.5:0.5b`) via HTTP sur le port 11434
            si tu veux une vraie réponse générée (totalement optionnel).

    Cas d'usage du gate :
    - "What does memo RX-7K3M-2H8P refer to?" → retrieval fort → réponse.
    - "Who won the World Cup in 2018?"        → retrieval faible → None.

    Pièges connus :
    - Threshold trop bas (< 0.1) : le gate ne se déclenche jamais, le système
      hallucine. Check `hallucination_gate_triggers` échoue.
    - Threshold trop haut (> 0.7) : le gate se déclenche TROP, le check
      `hallucination_gate_triggers` peut passer mais ton recall in-domain
      s'effondre car les vraies questions sont refusées. Les fixtures ont
      été calibrées pour qu'un seuil entre 0.25 et 0.5 fonctionne bien.
    """
    # TODO: appeler top_k_search(question, k=5, conn, model=model)
    # TODO: si len(hits) == 0 ou hits[0]["score"] < threshold:
    #           raise LowConfidenceRetrievalError(...)   # OU return None
    # TODO: sinon, construire une réponse (template ou Ollama) et la retourner.
    raise NotImplementedError(
        "generate_answer() pas encore implémenté. Branche top_k_search() et "
        "ajoute le hallucination gate sur hits[0]['score']."
    )


def main() -> None:
    """Demo : pose une question in-domain et une question hors-domaine."""
    from src.embed import load_model
    from src.store import get_engine

    engine = get_engine()
    model = load_model()
    with engine.connect() as conn:
        for q in [
            "Which support article mentions the reference RX-7K3M-2H8P?",
            "Will it rain in Paris tomorrow?",
        ]:
            try:
                ans = generate_answer(q, conn, model=model)
            except LowConfidenceRetrievalError as e:
                ans = None
                print(f"  Q: {q}\n  -> REFUSED ({e})\n")
                continue
            print(f"  Q: {q}\n  -> {ans!r}\n")


if __name__ == "__main__":
    main()
