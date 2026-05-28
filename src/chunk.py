"""Lumora Stock — document chunker (scaffold).

Ce module est volontairement INCOMPLET. À toi de l'implémenter en suivant
le contrat décrit dans `contracts/chunks.json` et le brief de `README.fr.md`.

Objectif final :

    python -m src.chunk

doit lire `fixtures/documents.jsonl` (5 000 documents) et produire en mémoire
une liste de chunks prête à être embeddée. La fonction `split_documents()`
doit pouvoir être appelée par `src.embed` et `src.store` sans relancer le
chunking deux fois.

CONTRATS À RESPECTER
- Préserver les codes mémo (`RX-7K3M-2H8P`) en TOKEN ENTIER dans le chunk.
  C'est la clé du retrieval : si ton chunker coupe un memo au milieu, ton
  recall s'effondre.
- Produire entre 30 000 et 100 000 chunks au total (sur 5 000 docs).
  En dehors de cette plage, le check `chunks_count_matches_contract` échoue.
  Repère : viser ~6-15 chunks/doc en moyenne.
- Garantir que chaque chunk contient le `doc_id` parent (sinon on ne peut
  pas mesurer le recall).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
DOCS_FILE = FIXTURES_DIR / "documents.jsonl"


@dataclass(frozen=True)
class Chunk:
    """Structure minimale d'un chunk avant embedding.

    `tokens` est ESTIMÉ — un word-count ou un tokenizer-based count fait
    l'affaire. C'est un garde-fou d'observabilité, pas une mesure exacte.
    """
    doc_id: int
    chunk_text: str
    tokens: int


def iter_documents(path: Path = DOCS_FILE) -> Iterator[dict]:
    """Itère sur le corpus JSONL. Helper fourni — pas à modifier."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_documents(docs: Iterable[dict]) -> list[Chunk]:
    """Découpe une liste de documents en chunks prêts à être embeddés.

    Stratégies acceptées (à toi de choisir et de justifier dans le README) :

    - **Fixed-size avec overlap** : par ex. 200 caractères avec 50 d'overlap.
      Simple, déterministe, robuste — mais peut couper au milieu d'un memo
      si tu ne fais pas attention. Solution : split par phrases d'abord,
      puis re-batch par taille.

    - **Sentence-based** : split sur les retours-ligne et les '. ', puis
      regroupe les phrases courtes. Préserve naturellement les memo codes.
      C'est probablement ton meilleur choix sur ce corpus.

    - **Semantic chunking** : trop lourd pour ce projet. Skip.

    Le plus important : ne JAMAIS couper un token de la forme `XX-XXXX-XXXX`
    en deux. Tu peux le vérifier rapidement avec une regex pendant les tests
    en local.

    Args:
        docs: itérable de dicts {"doc_id", "title", "body", "category", "memo_code"}

    Returns:
        Liste de Chunk. L'ordre est libre mais doit être déterministe (sinon
        l'INSERT n'est pas idempotent dans certains cas).
    """
    # TODO: pour chaque document, construire le texte source (titre + body)
    # TODO: découper en chunks selon ta stratégie
    # TODO: calculer un `tokens` approximatif (len(text.split()) est ok)
    # TODO: retourner List[Chunk] avec doc_id propagé
    raise NotImplementedError(
        "split_documents() pas encore implémenté. Choisis une stratégie de "
        "chunking et garde les memo codes intacts. Vise 30k-100k chunks au total."
    )


def main() -> None:
    """Lance le chunking en standalone — utile pour debugger en local."""
    docs = list(iter_documents())
    chunks = split_documents(docs)
    print(f"loaded {len(docs)} documents")
    print(f"produced {len(chunks)} chunks "
          f"(avg {len(chunks) / max(len(docs), 1):.1f} chunks/doc)")


if __name__ == "__main__":
    main()
