"""IAmDataEng — rubric d'évaluation pour `transformation.rag-pipeline-pgvector`.

Six checks déterministes alignés sur `contracts/chunks.json` et la spec projet.
Chaque check produit, en cas d'échec, un message pédagogique en français qui
pointe la cause probable. Ce module est lancé tel quel par le workflow
`.github/workflows/iamdataeng-evaluate.yml` et par le learner en local via
`pytest tests/`.

Les tests s'appuient SUR L'ARTEFACT (la table `chunks` dans Postgres) après
exécution du pipeline complet. C'est volontaire : on évalue le résultat, pas
le style. La fixture `pipeline_executed` est `scope="session"` pour qu'on ne
relance le pipeline qu'UNE fois — il prend ~2 minutes pour embedder 50k chunks
sur le runner GitHub Actions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = PROJECT_ROOT / "contracts" / "chunks.json"
GOLDEN_PATH = PROJECT_ROOT / "fixtures" / "golden_eval.json"
OOD_PATH = PROJECT_ROOT / "fixtures" / "out_of_domain.json"
DOCS_PATH = PROJECT_ROOT / "fixtures" / "documents.jsonl"

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://rag:rag@localhost:5432/rag"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_fixtures() -> None:
    """Régénère les fixtures si elles ne sont pas là.

    Les fixtures sont normalement créées par `.devcontainer/post-create.sh`
    et par l'étape CI dédiée. Cette fonction est un filet de sécurité.
    """
    if DOCS_PATH.exists() and GOLDEN_PATH.exists() and OOD_PATH.exists():
        return
    result = subprocess.run(
        [sys.executable, "-m", "fixtures.generate_fixtures"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "Les fixtures n'ont pas pu être régénérées.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _run_full_pipeline() -> None:
    """Exécute le pipeline complet : `python -m src.store`.

    `src.store.main()` est le point d'entrée intégré : il appelle chunk →
    embed → upsert. C'est ce qu'un learner lancerait en prod.
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.store"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if result.returncode != 0:
        pytest.fail(
            "Le pipeline `python -m src.store` a planté.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            "Astuce : implémente dans l'ordre src/chunk.py → src/embed.py "
            "→ src/store.py — tous lèvent NotImplementedError par défaut."
        )


# ---------------------------------------------------------------------------
# Fixture session : pipeline lancé UNE seule fois
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_executed():
    """Lance le pipeline complet une fois et donne un engine SQLAlchemy."""
    _ensure_fixtures()
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

    # On purge la table avant chaque session de tests pour partir d'un état
    # déterministe. Si la table n'existe pas encore (premier run), on ignore.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE;"))

    _run_full_pipeline()
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Check 1 — pgvector_table_schema_matches
# ---------------------------------------------------------------------------


def test_pgvector_table_schema_matches(pipeline_executed):
    """La table `chunks` existe avec colonnes + types EXACTEMENT comme le contrat."""
    engine = pipeline_executed
    contract = _load_contract()
    expected = {c["name"]: c for c in contract["columns"]}

    with engine.connect() as conn:
        # information_schema.columns retourne data_type en string ('text',
        # 'bigint', 'integer', 'timestamp with time zone', 'USER-DEFINED'
        # pour vector — qu'on désambiguïse via udt_name).
        rows = conn.execute(text("""
            SELECT column_name, data_type, udt_name, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'chunks'
            ORDER BY ordinal_position
        """)).fetchall()

    if not rows:
        pytest.fail(
            "La table `chunks` n'existe pas dans Postgres.\n"
            "Vérifie que `init_schema()` est appelé dans src/store.main() et "
            "qu'il fait bien `CREATE TABLE chunks (...)`."
        )

    actual_cols = {r[0] for r in rows}
    expected_cols = set(expected)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing or extra:
        msg = ["Le schéma de la table `chunks` ne correspond pas au contrat."]
        if missing:
            msg.append(f"Colonnes manquantes : {sorted(missing)}")
        if extra:
            msg.append(f"Colonnes en trop    : {sorted(extra)}")
        msg.append("Relis contracts/chunks.json — les noms doivent être EXACTS.")
        pytest.fail("\n".join(msg))

    # Type checking, colonne par colonne.
    type_errors = []
    for col_name, data_type, udt_name, is_nullable in rows:
        spec = expected[col_name]
        # data_type lower-case attendu
        wanted = spec["pg_type"].lower()
        ok = False
        if wanted == "vector":
            ok = (udt_name == "vector")
        elif wanted == "text":
            ok = (data_type == "text")
        elif wanted == "bigint":
            ok = (data_type == "bigint")
        elif wanted == "integer":
            ok = (data_type == "integer")
        elif wanted == "timestamp with time zone":
            ok = (data_type == "timestamp with time zone")
        else:
            ok = (data_type == wanted)
        if not ok:
            type_errors.append((col_name, wanted, data_type, udt_name))

        # Nullability
        actual_null = (is_nullable == "YES")
        if actual_null != spec["nullable"]:
            type_errors.append((
                col_name,
                f"nullable={spec['nullable']}",
                f"nullable={actual_null}",
                "",
            ))

    if type_errors:
        lines = ["Types Postgres incorrects (col, attendu, obtenu, udt_name) :"]
        lines += [f"  {e}" for e in type_errors]
        lines.append(
            "Indices :\n"
            "  - embedding doit être VECTOR(384) — udt_name='vector'\n"
            "  - created_at doit être TIMESTAMPTZ (`timestamp with time zone`)\n"
            "  - id et doc_id doivent être BIGINT\n"
            "  - tokens doit être INTEGER (pas BIGINT)"
        )
        pytest.fail("\n".join(lines))

    # Bonus check spécifique : la dimension du vecteur. atttypmod encode la dim.
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = 'chunks' AND a.attname = 'embedding'
        """)).fetchone()
    if row is not None:
        dim = row[0]  # pgvector encodes dim directly in atttypmod
        expected_dim = expected["embedding"].get("vector_dim", 384)
        if dim != expected_dim:
            pytest.fail(
                f"La colonne `embedding` a la dimension {dim}, attendue {expected_dim}.\n"
                f"Le modèle pinné `all-MiniLM-L6-v2` produit des vecteurs 384-dim. "
                f"Crée la colonne en `vector({expected_dim})`."
            )


# ---------------------------------------------------------------------------
# Check 2 — chunks_count_matches_contract
# ---------------------------------------------------------------------------


def test_chunks_count_matches_contract(pipeline_executed):
    """Le COUNT(chunks) doit être dans [30_000, 100_000].

    Cette plage encode une discipline de chunking :
    - < 30k chunks sur 5k docs → < 6 chunks/doc → chunking trop grossier, le
      contexte d'un chunk est trop large pour un retrieval fin.
    - > 100k chunks sur 5k docs → > 20 chunks/doc → chunking trop fin, on
      perd le contexte sémantique et on multiplie les coûts de stockage et
      d'indexation pour rien.
    """
    engine = pipeline_executed
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
    if n < 30_000:
        pytest.fail(
            f"Seulement {n} chunks dans la table — trop peu (seuil mini 30 000).\n"
            "Ton chunker est trop grossier : tu produis < 6 chunks par document.\n"
            "Indice : un chunking par phrase, ou fixed-size 200-300 chars avec\n"
            "overlap, te place naturellement dans la bonne plage."
        )
    if n > 100_000:
        pytest.fail(
            f"{n} chunks dans la table — trop nombreux (seuil maxi 100 000).\n"
            "Ton chunker est trop fin : tu produis > 20 chunks par document.\n"
            "Chaque chunk perd alors son contexte sémantique. Augmente la\n"
            "taille cible ou réduis l'overlap."
        )


# ---------------------------------------------------------------------------
# Check 3 — embeddings_deterministic
# ---------------------------------------------------------------------------


def test_embeddings_deterministic(pipeline_executed):
    """Embedder deux fois le même texte produit le MÊME vecteur, bit-pour-bit.

    On charge le modèle, on embed le même texte deux fois, on compare. Si
    le learner a accidentellement introduit du non-déterminisme (random
    sampling, dropout, GPU non-déterministe), ce test échoue.
    """
    # On importe ici pour ne pas payer 2-3s de chargement avant que la
    # première fixture ait montré une erreur évidente.
    from src.embed import embed_chunks, load_model

    model = load_model()
    sample = [
        "Lumora Inventory: reorder rules trigger when SKU crosses threshold.",
        "Lumora Inventory: reorder rules trigger when SKU crosses threshold.",
    ]
    try:
        v1 = embed_chunks([sample[0]], model=model)
        v2 = embed_chunks([sample[1]], model=model)
    except NotImplementedError as e:
        pytest.fail(
            "embed_chunks() lève NotImplementedError. Implémente-la dans "
            f"src/embed.py.\n  Détail : {e}"
        )

    if v1.shape != v2.shape:
        pytest.fail(
            f"Shapes différents pour deux runs du même input : {v1.shape} vs {v2.shape}.\n"
            "Vérifie que embed_chunks() retourne toujours (n, 384) avec n = len(input)."
        )

    if not np.array_equal(v1, v2):
        max_diff = float(np.max(np.abs(v1 - v2)))
        pytest.fail(
            f"Les embeddings ne sont PAS déterministes — max diff = {max_diff:.6e}.\n"
            "Causes possibles :\n"
            "  - tu as activé un dropout (sentence-transformers est en .eval() par défaut)\n"
            "  - tu utilises un GPU avec cuDNN non-déterministe (utilise CPU en CI)\n"
            "  - tu shuffles les inputs avant encoding (l'ordre OUT doit suivre l'ordre IN)\n"
            "  - tu utilises un modèle non-pinné (vérifie EMBEDDING_MODEL=all-MiniLM-L6-v2)"
        )

    # Bonus : vérifier la normalisation L2
    norm = float(np.linalg.norm(v1[0]))
    if not (0.99 < norm < 1.01):
        pytest.fail(
            f"Les vecteurs ne sont pas L2-normalisés (norme = {norm:.4f}, attendue ~1.0).\n"
            "Ajoute `normalize_embeddings=True` à model.encode(...). Sans ça, la\n"
            "similarité cosine pgvector (<=>) donnera des résultats incohérents."
        )


# ---------------------------------------------------------------------------
# Checks 4 & 5 — retrieval_recall_at_5 et retrieval_mrr
# On lance l'éval UNE fois, on partage les métriques entre les deux tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def retrieval_metrics(pipeline_executed):
    """Lance l'évaluation complète et retourne (results, metrics).

    Imports retardés pour ne payer le coût qu'après les checks structurels.
    """
    from src.embed import load_model
    from src.evaluate import explain_failures, run_evaluation
    from src.retrieve import top_k_search

    engine = pipeline_executed
    model = load_model()
    with engine.connect() as conn:
        try:
            results, metrics = run_evaluation(top_k_search, conn, model, k=10)
        except NotImplementedError as e:
            pytest.fail(
                f"top_k_search() lève NotImplementedError. Implémente-la dans "
                f"src/retrieve.py.\n  Détail : {e}"
            )
    return results, metrics, explain_failures


def test_retrieval_recall_at_5(retrieval_metrics):
    """Recall@5 ≥ 0.70 sur les 50 questions golden.

    Sur 50 questions où la réponse est encodée comme un memo code unique dans
    un seul chunk, un pipeline correctement réglé atteint typiquement
    recall@5 ≥ 0.90. Le seuil 0.70 laisse de la marge pour des chunkers
    sous-optimaux mais sanctionne les pipelines vraiment cassés.
    """
    results, metrics, explain = retrieval_metrics
    if metrics["recall_at_k"] < 0.70:
        pytest.fail(
            f"recall@5 = {metrics['recall_at_k']:.3f} (seuil 0.70).\n"
            f"{explain(results, k=5)}\n"
            "Causes probables :\n"
            "  - chunker qui coupe les memo codes en deux (vérifie qu'ils sont\n"
            "    présents en entier dans chunk_text)\n"
            "  - embeddings non-normalisés (le cosine pgvector devient bruité)\n"
            "  - mauvais opérateur SQL (<-> vs <=> donnent des rangs différents)\n"
            "  - mauvais modèle utilisé entre store et retrieve\n"
            "  - LIMIT absent ou trop bas dans la requête"
        )


def test_retrieval_mrr(retrieval_metrics):
    """MRR ≥ 0.45.

    Un MRR de 0.45 correspond grossièrement à « le bon doc apparaît en moyenne
    en position 2-3 ». Plus bas que ça, ton ranking est cassé même si ton
    recall est bon — ce qui est inacceptable pour un assistant qui ne lit
    que le top-1.
    """
    results, metrics, explain = retrieval_metrics
    if metrics["mrr"] < 0.45:
        pytest.fail(
            f"MRR = {metrics['mrr']:.3f} (seuil 0.45).\n"
            f"recall@5 = {metrics['recall_at_k']:.3f}\n"
            "Le bon doc apparaît dans le top-k mais trop loin dans le ranking.\n"
            "Pistes :\n"
            "  - vérifier que les vecteurs sont bien L2-normalisés AVANT insert ET avant query\n"
            "  - vérifier que le modèle store == le modèle retrieve\n"
            "  - utiliser cosine (<=>) plutôt que L2 (<->) — sur des vecteurs\n"
            "    normalisés, les deux ordonnent identiquement, mais sur des\n"
            "    vecteurs non-normalisés, L2 favorise les vecteurs courts\n"
            "  - ton chunking sépare peut-être le memo code du contexte sémantique."
        )


# ---------------------------------------------------------------------------
# Check 6 — hallucination_gate_triggers
# ---------------------------------------------------------------------------


def test_hallucination_gate_triggers(pipeline_executed):
    """Les 10 questions hors-domaine doivent toutes déclencher le gate.

    Le gate est OK si `generate_answer()` retourne None OU lève
    `LowConfidenceRetrievalError`. Tout autre retour (string non vide) est
    considéré comme une hallucination.

    On exige 9/10 minimum pour tolérer une question borderline (par ex. si
    le learner a un seuil un peu généreux). En-dessous, le système est
    laissé en mode « réponds toujours » et hallucine en prod.
    """
    from src.answer import LowConfidenceRetrievalError, generate_answer
    from src.embed import load_model

    engine = pipeline_executed
    with OOD_PATH.open("r", encoding="utf-8") as f:
        ood_questions = json.load(f)

    model = load_model()
    refused = 0
    answered = []
    with engine.connect() as conn:
        for item in ood_questions:
            q = item["question"]
            try:
                ans = generate_answer(q, conn, model=model)
            except LowConfidenceRetrievalError:
                refused += 1
                continue
            except NotImplementedError as e:
                pytest.fail(
                    "generate_answer() lève NotImplementedError. Implémente-la "
                    f"dans src/answer.py.\n  Détail : {e}"
                )
            if ans is None or (isinstance(ans, str) and ans.strip() == ""):
                refused += 1
            else:
                answered.append((q, ans[:80]))

    n = len(ood_questions)
    if refused < 9:
        examples = "\n".join(f"  Q={q!r}\n    -> {a!r}..." for q, a in answered[:3])
        pytest.fail(
            f"Le hallucination gate ne s'est déclenché que sur {refused}/{n} questions "
            "hors-domaine (seuil mini : 9/10).\n"
            "Ton seuil de confiance est probablement trop bas, OU tu ne checkes pas du "
            "tout le score du top-1 avant de répondre.\n"
            f"Exemples de questions où le gate a échoué :\n{examples}\n"
            "Pistes :\n"
            "  - vérifier que tu lis bien `hits[0]['score']` ET que tu compares\n"
            "    au bon seuil (par défaut 0.35).\n"
            "  - si tu retournes toujours quelque chose 'au cas où', SUPPRIME ce\n"
            "    comportement. Refuser de répondre est un feature, pas un bug."
        )
