"""Lumora Stock — retrieval evaluation helpers.

CE MODULE EST FOURNI EN CLAIR. Tu n'as PAS à le réécrire.

Pourquoi ? Les métriques recall@k et MRR sont du code de 10 lignes — les
réimplémenter ferait perdre 2 heures au learner sans apprentissage data
engineering significatif. Ce qui compte ici, c'est :
- savoir lire ces métriques,
- comprendre ce qu'elles disent (et ne disent pas),
- les utiliser comme garde-fou dans une CI.

Recall@k    = part des questions où le bon doc apparaît dans le top-k.
MRR (Mean Reciprocal Rank) = moyenne de 1/rang_du_bon_doc, 0 si absent.

Recall mesure « le bon doc est-il là quelque part dans le top-k ? »
MRR mesure « à quel point le bon doc est haut dans le ranking ? »

Une rubric senior exige les deux : un recall@5 de 0.95 avec un MRR de 0.20
veut dire que tu trouves le bon doc, mais toujours en 4ᵉ position. C'est
inacceptable pour un assistant qui ne lit que le top-1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "fixtures" / "golden_eval.json"


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    expected_doc_id: int
    retrieved_doc_ids: list[int]
    rank_of_expected: int | None  # 1-based, or None if not in top-k


def load_golden() -> list[dict]:
    """Charge le fichier golden_eval.json — 50 questions in-domain."""
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def rank_of_expected(retrieved_doc_ids: list[int], expected_doc_id: int) -> int | None:
    """Retourne le rang 1-based du bon doc, ou None s'il n'est pas dans la liste.

    On dédoublonne d'abord les `retrieved_doc_ids` car plusieurs chunks
    peuvent appartenir au MÊME doc. Le rang est calculé sur l'ordre de
    première apparition de chaque doc unique.
    """
    seen = []
    for did in retrieved_doc_ids:
        if did not in seen:
            seen.append(did)
        if did == expected_doc_id:
            return len(seen)  # 1-based
    return None


def recall_at_k(results: list[RetrievalResult], k: int) -> float:
    """Recall@k = part des questions où le bon doc est dans le top-k.

    `rank_of_expected` est calculé sur la liste des doc_ids retournés par
    le learner ; on considère qu'un doc est « trouvé » si son rang ≤ k.
    """
    if not results:
        return 0.0
    found = sum(
        1 for r in results
        if r.rank_of_expected is not None and r.rank_of_expected <= k
    )
    return found / len(results)


def mean_reciprocal_rank(results: list[RetrievalResult]) -> float:
    """MRR = moyenne de 1/rang_du_bon_doc, avec 0 si le doc est absent.

    Plus le bon doc est haut, plus le MRR est élevé.
    - rang 1 → 1.0
    - rang 2 → 0.5
    - rang 5 → 0.2
    - absent → 0.0
    """
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.rank_of_expected is not None and r.rank_of_expected > 0:
            total += 1.0 / r.rank_of_expected
    return total / len(results)


def run_evaluation(
    top_k_search_fn,
    conn,
    model,
    k: int = 10,
    golden: list[dict] | None = None,
) -> tuple[list[RetrievalResult], dict]:
    """Lance la rubric d'évaluation complète.

    Args:
        top_k_search_fn: la fonction `src.retrieve.top_k_search` du learner.
        conn: connexion SQLAlchemy active.
        model: SentenceTransformer pré-chargé (passé pour éviter de le
            recharger 50 fois).
        k: profondeur du top-k passée au search. Doit être ≥ 5 pour calculer
            recall@5.
        golden: golden set chargé en amont si tu veux le filtrer ; sinon
            charge le fichier par défaut.

    Returns:
        (per-question RetrievalResults, agrégats {recall@5, mrr, k, n_queries}).
    """
    if golden is None:
        golden = load_golden()

    results: list[RetrievalResult] = []
    for item in golden:
        hits = top_k_search_fn(item["question"], k=k, conn=conn, model=model)
        retrieved_ids = [h["doc_id"] for h in hits]
        results.append(RetrievalResult(
            question=item["question"],
            expected_doc_id=item["expected_doc_id"],
            retrieved_doc_ids=retrieved_ids,
            rank_of_expected=rank_of_expected(retrieved_ids, item["expected_doc_id"]),
        ))

    metrics = {
        "recall_at_k": recall_at_k(results, k=5),
        "mrr": mean_reciprocal_rank(results),
        "k": k,
        "n_queries": len(results),
    }
    return results, metrics


def explain_failures(results: list[RetrievalResult], k: int = 5) -> str:
    """Joli rapport pour debug — où le retrieval s'est planté."""
    misses = [r for r in results
              if r.rank_of_expected is None or r.rank_of_expected > k]
    if not misses:
        return f"All {len(results)} queries hit within top-{k}."
    lines = [f"{len(misses)}/{len(results)} queries MISSED top-{k}:"]
    for r in misses[:10]:
        rank = r.rank_of_expected if r.rank_of_expected else f">top-{k}"
        lines.append(f"  rank={rank}  expected={r.expected_doc_id}  Q={r.question!r}")
    if len(misses) > 10:
        lines.append(f"  ... and {len(misses) - 10} more.")
    return "\n".join(lines)
