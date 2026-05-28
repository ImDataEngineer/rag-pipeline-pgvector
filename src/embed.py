"""Lumora Stock — embedder (scaffold).

Encode les chunks produits par `src.chunk` en vecteurs 384-dim avec le modèle
local `sentence-transformers/all-MiniLM-L6-v2` (pinné pour le déterminisme).

CONTRATS À RESPECTER
- Vecteurs L2-normalisés (sinon cosine vs inner product divergent).
- Encoding DÉTERMINISTE : embedder deux fois le MÊME texte doit produire le
  MÊME vecteur, bit-pour-bit, sur la même machine. C'est le check
  `embeddings_deterministic`. sentence-transformers est déterministe sur CPU
  tant que tu n'actives pas de dropout — par défaut c'est OK.
- Pas d'appel OpenAI/Anthropic/Cohere. Tout est LOCAL, tout est CPU,
  rien ne sort de la machine.

Indice perf : `model.encode(texts, batch_size=64)` est ~10x plus rapide que
de boucler chunk par chunk. Sur 50k chunks tu veux du batching.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384


def load_model(model_name: str = DEFAULT_MODEL):
    """Charge le SentenceTransformer. Helper fourni — pas à modifier.

    Le modèle est pré-téléchargé par `.devcontainer/post-create.sh` et caché
    dans `$HF_HOME`. Premier appel : ~1s. Suivants : ~50ms.
    """
    # Import retardé : `sentence-transformers` met 2-3s à s'importer et on
    # n'en veut pas l'overhead si l'appelant n'embed pas réellement.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_chunks(texts: Iterable[str], model=None) -> np.ndarray:
    """Encode une séquence de textes en matrice (n, 384) L2-normalisée.

    Args:
        texts: itérable de strings à embedder.
        model: SentenceTransformer optionnel. Si None, on en charge un par défaut.

    Returns:
        np.ndarray de shape (len(texts), 384), dtype float32, L2-normalisé.

    Pièges connus :
    - Si tu oublies `normalize_embeddings=True`, ta similarité cosine sera
      faussée — pgvector utilise `<=>` pour cosine et `<#>` pour negative
      inner-product. Avec des vecteurs L2-normalisés, les deux sont
      équivalents (et plus rapides).
    - Si tu shuffles l'ordre d'entrée, tu casses l'alignement avec les chunks.
      L'ordre OUT doit correspondre à l'ordre IN.
    - Sur GPU, sentence-transformers peut être non-déterministe (cuDNN). En CI
      on est en CPU, donc OK — mais le check `embeddings_deterministic` te
      protégera si tu changes ça plus tard.
    """
    # TODO: charger le model si None
    # TODO: model.encode(list(texts), batch_size=..., normalize_embeddings=True,
    #                    convert_to_numpy=True, show_progress_bar=False)
    # TODO: vérifier shape (n, 384) et dtype float32 avant de retourner
    raise NotImplementedError(
        "embed_chunks() pas encore implémenté. Utilise SentenceTransformer.encode "
        "avec normalize_embeddings=True et convert_to_numpy=True."
    )


def main() -> None:
    """Smoke test : embed 3 textes et imprime leurs normes."""
    model = load_model()
    sample = [
        "Reorder rules in Lumora Inventory.",
        "Reorder rules in Lumora Inventory.",  # doublon pour vérifier déterminisme
        "Comment configurer une cycle count?",
    ]
    vecs = embed_chunks(sample, model=model)
    print(f"shape = {vecs.shape}, dtype = {vecs.dtype}")
    print(f"norms = {np.linalg.norm(vecs, axis=1)}  (doit valoir ~1.0 chacune)")
    print(f"same? {np.allclose(vecs[0], vecs[1])}  (doit valoir True)")


if __name__ == "__main__":
    main()
