"""Lumora Stock — pgvector store (scaffold).

Crée la table `chunks` conforme à `contracts/chunks.json` et upsert les
chunks embeddés dans Postgres + pgvector.

CONTRATS À RESPECTER
- Schéma EXACT (cf. contracts/chunks.json) :
    id BIGINT PK,
    doc_id BIGINT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    tokens INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
- L'extension `vector` DOIT être activée (`CREATE EXTENSION IF NOT EXISTS vector`).
- L'INSERT est IDEMPOTENT : relancer le pipeline ne double pas la table.
  Deux stratégies acceptées :
    1. TRUNCATE chunks avant chaque run (simple, lent sur 50k mais OK).
    2. UPSERT sur une clé naturelle (doc_id, hash(chunk_text)) avec
       ON CONFLICT DO NOTHING. Plus subtil, plus utile en prod.

INDEX (RECOMMANDÉ, PAS OBLIGATOIRE)
- `CREATE INDEX chunks_embedding_ivfflat_idx ON chunks USING ivfflat
   (embedding vector_cosine_ops) WITH (lists = 100);`
   ou
- `CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw
   (embedding vector_cosine_ops);`
- ivfflat est plus rapide à construire, hnsw est plus précis sur de petits
  corpus. Sur 50k chunks, les deux marchent — fais ton choix et défends-le.
- Sans index, pgvector fait un sequential scan : OK pour 50k, catastrophe
  pour 5M.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from sqlalchemy import Connection, create_engine

from src.chunk import Chunk

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://rag:rag@localhost:5432/rag"
)


def get_engine():
    """Helper fourni — engine SQLAlchemy avec pool minimal."""
    return create_engine(DATABASE_URL, pool_size=2, max_overflow=2, future=True)


def init_schema(conn: Connection) -> None:
    """Crée l'extension pgvector + la table `chunks` (+ index optionnel).

    DOIT être idempotent : `CREATE EXTENSION IF NOT EXISTS`, `CREATE TABLE
    IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`.

    Pièges :
    - `VECTOR(384)` : la dimension est fixée par le modèle. Si tu changes
      d'embedder un jour, tu DROP la table — pgvector ne supporte pas le
      resize d'une colonne VECTOR.
    - `created_at TIMESTAMPTZ DEFAULT now()` : pas `TIMESTAMP` simple. Une
      data warehouse sans timezone, c'est un piège qu'on paie 6 mois plus tard.
    """
    # TODO: CREATE EXTENSION IF NOT EXISTS vector;
    # TODO: CREATE TABLE IF NOT EXISTS chunks (...)
    # TODO (recommandé): CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    raise NotImplementedError(
        "init_schema() pas encore implémenté. Suis contracts/chunks.json à la lettre."
    )


def upsert_to_pgvector(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    conn: Connection,
) -> int:
    """Upsert une batch de chunks + vecteurs dans Postgres + pgvector.

    Args:
        chunks: liste de Chunk (taille n).
        embeddings: matrice (n, 384) L2-normalisée.
        conn: connexion SQLAlchemy active (transaction gérée par l'appelant).

    Returns:
        Nombre de lignes effectivement insérées.

    Stratégie idempotente attendue (au choix) :
    - TRUNCATE chunks; puis INSERT en batch (executemany / psycopg2.extras.execute_values).
    - INSERT ... ON CONFLICT (doc_id, content_hash) DO NOTHING — ajouter alors
      un UNIQUE INDEX dans init_schema().

    Perf : sur 50k chunks, batcher par 500-1000 lignes. Les single-row INSERT
    sont 100x plus lents.

    Format vecteur côté Postgres : `pgvector` (Python) sait sérialiser les
    np.ndarray via le type `Vector(384)` SQLAlchemy. Sinon, le fallback est
    de passer la liste sous forme `'[0.1,0.2,...]'::vector`.
    """
    # TODO: prepare values (id auto, doc_id, chunk_text, embedding, tokens)
    # TODO: bulk insert via executemany ou execute_values
    # TODO: retourner le rowcount
    raise NotImplementedError(
        "upsert_to_pgvector() pas encore implémenté. Pense bulk + idempotence."
    )


def main() -> None:
    """Pipeline complet store : chunk → embed → upsert."""
    from src.chunk import iter_documents, split_documents
    from src.embed import embed_chunks, load_model

    engine = get_engine()
    with engine.begin() as conn:
        init_schema(conn)

    docs = list(iter_documents())
    chunks = split_documents(docs)
    model = load_model()
    texts = [c.chunk_text for c in chunks]
    embeddings = embed_chunks(texts, model=model)

    with engine.begin() as conn:
        n = upsert_to_pgvector(chunks, embeddings, conn)
    print(f"upserted {n} chunks into pgvector")


if __name__ == "__main__":
    main()
