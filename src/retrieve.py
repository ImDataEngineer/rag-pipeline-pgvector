"""Lumora Stock — top-k retrieval (scaffold).

Pour une question utilisateur, retourne les `k` chunks les plus pertinents
selon la similarité cosinus calculée par pgvector.

CONTRATS À RESPECTER
- La fonction retourne une LISTE de dicts avec AU MINIMUM les clés
  `doc_id`, `chunk_text`, `score`. Le test suite lit ces champs.
- L'ordre est trié par pertinence décroissante (meilleur score en premier).
- `score` est un flottant dans [0.0, 1.0] où 1.0 = identique, 0.0 = orthogonal.
  Si tu utilises l'opérateur `<=>` de pgvector (cosine distance), la
  similarité est `1.0 - distance`.

RAPPEL pgvector :
- `<=>` : cosine distance        (0 = identique, 2 = opposé)
- `<#>` : negative inner product (équivalent à cosine si vecteurs normalisés)
- `<->` : L2 distance            (à éviter ici, ne donne pas la même chose)

Le check `retrieval_recall_at_5` veut k=5, et `retrieval_mrr` veut k=10
(pour avoir une décroissance lisible). Ta fonction doit gérer un k variable.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from sqlalchemy import Connection, create_engine, text

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://rag:rag@localhost:5432/rag"
)


def get_engine():
    """Helper fourni."""
    return create_engine(DATABASE_URL, pool_size=2, max_overflow=2, future=True)


def top_k_search(
    query: str,
    k: int,
    conn: Connection,
    model=None,
) -> list[dict]:
    """Embed la question et retourne les `k` chunks les plus proches.

    Args:
        query: texte de la question utilisateur.
        k: nombre de chunks à retourner (≥ 1).
        conn: connexion SQLAlchemy active.
        model: SentenceTransformer optionnel. Si None, on en charge un.

    Returns:
        Liste de dicts triés par score décroissant, chacun avec au moins :
            { "doc_id": int, "chunk_text": str, "score": float }

    Pièges connus :
    - Embedder la question avec un autre modèle que le store. Garde le même
      modèle, même version, même `normalize_embeddings=True`.
    - Oublier l'ORDER BY : pgvector ne trie pas tout seul.
    - Demander k=5 mais retourner 50 (LIMIT manquant) : tes tests passent
      mais ta latence prod explose.
    - Mélanger cosine et L2 : `<=>` (cosine) et `<->` (L2) renvoient des
      classements DIFFÉRENTS si les vecteurs ne sont pas normalisés.

    Format de la requête (à adapter) :

        SELECT doc_id, chunk_text, 1.0 - (embedding <=> :q) AS score
        FROM chunks
        ORDER BY embedding <=> :q
        LIMIT :k

    `:q` se passe en string `'[0.1,0.2,...]'` ou via le type pgvector.Vector.
    """
    # TODO: embed la query (1 vecteur 384-dim normalisé)
    # TODO: convertir le vecteur en littéral pgvector ou utiliser le type Vector
    # TODO: exécuter la requête SQL ci-dessus avec ORDER BY + LIMIT
    # TODO: retourner [{"doc_id": ..., "chunk_text": ..., "score": ...}, ...]
    raise NotImplementedError(
        "top_k_search() pas encore implémenté. Embed la query, puis ORDER BY "
        "embedding <=> :q LIMIT :k."
    )


def main() -> None:
    """Demo : lance une recherche sur une question d'exemple."""
    from src.embed import load_model

    engine = get_engine()
    model = load_model()
    sample = "Which support article mentions the reference RX-7K3M-2H8P?"
    with engine.connect() as conn:
        hits = top_k_search(sample, k=5, conn=conn, model=model)
    for i, hit in enumerate(hits, start=1):
        snippet = hit["chunk_text"][:120].replace("\n", " ")
        print(f"{i}. doc_id={hit['doc_id']}  score={hit['score']:.4f}  {snippet}")


if __name__ == "__main__":
    main()
