# A production-grade RAG pipeline, measured — `transformation.rag-pipeline-pgvector`

> **Level**: senior · **Estimated time**: ~12 h · **Price**: €49
> **Framework axis**: `transformation` (sub-skills: `data_modeling`,
> `incremental_transforms`, `performance_tuning`)
> **Secondary axes covered**: `storage` (pgvector + ANN index),
> `software_engineering_dataops` (determinism, rigorous evaluation).

**A production RAG pipeline is 80% DATA and 20% LLM. This project tests
the 80%.**

If you want to learn how to "call GPT-4 with some context", you're in
the wrong place. If you want to learn how to build the pipeline that,
on the day your LLM hallucinates, lets you prove it was the LLM's fault
and not your data's — welcome.

The rubric does **not** grade the quality of generated answers. It
grades the quality of the pipeline: determinism, reproducibility,
observability, rigorous evaluation. That's what a senior tech
interviewer probes in a 2026 RAG interview — and what most candidates
fail.

---

## The context

Lumora Stock is a B2B inventory-management SaaS. Their support team has
accumulated **5,000 articles** (FAQ, troubleshooting, how-to,
changelogs) over four years. The team loses 4 hours a day manually
searching for answers that are already documented.

Your job: build the pipeline that ingests these articles, chunks them
cleanly, embeds them reproducibly, stores them in Postgres + pgvector,
answers a question with **measurable** retrieval (recall@5, MRR), and
**refuses to answer** when retrieval is too weak — so it does NOT
hallucinate on out-of-domain questions.

Stack:
- **Postgres 16 + pgvector** (`pgvector/pgvector:pg16`)
- **Python 3.11** + sqlalchemy + psycopg2 + numpy
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384 dims, local CPU)
- **pytest** for the rubric
- *(optional)* **Ollama** + `qwen2.5:0.5b` if you want to wire up real
  generation — the rubric does NOT need it

No OpenAI, no Anthropic, no API key. Everything runs locally, everything
is deterministic, everything is replayable in CI.

---

## What you ship

| Deliverable | Where |
|---|---|
| The chunker | `src/chunk.py` (`split_documents()`) |
| The embedder | `src/embed.py` (`embed_chunks()`) |
| The pgvector upsert | `src/store.py` (`init_schema()`, `upsert_to_pgvector()`) |
| Top-k search | `src/retrieve.py` (`top_k_search()`) |
| The hallucination gate | `src/answer.py` (`generate_answer()`) |
| An ADR (Architecture Decision Record) | `docs/decisions.md` — chunking choice, confidence threshold, ANN index if any |

`src/evaluate.py` **is shipped in plain form** (helpers `recall_at_k`,
`MRR`). You don't have to rewrite it — the senior rubric is on the
pipeline, not on the metric formulas.

---

## Getting started

In Codespaces (one-click open from the IAmDataEng app), the post-create
installs dependencies, starts Postgres+pgvector, pre-downloads the
sentence-transformers model, and generates the fixtures. So you only
have to code.

Locally:

```bash
# 1. Start Postgres + pgvector
docker compose -f .devcontainer/docker-compose.yml up -d postgres

# 2. Install dependencies (CPU-only torch to stay light)
pip install -r requirements.txt

# 3. Generate fixtures (deterministic — seed=42, ~5000 docs)
python -m fixtures.generate_fixtures

# 4. Implement src/*.py then run the pipeline
python -m src.store

# 5. Run the rubric
pytest tests/ -v
```

Once the 6 checks pass locally, **commit + push** to your fork. GitHub
Actions CI replays the same rubric (with an ephemeral Postgres+pgvector
service container).

---

## The 6 rubric checks

Defined in `tests/test_evaluate.py`. All **deterministic**. All read
**artifacts** (the Postgres table, the learner's Python functions) — not
code style.

| # | Check | What we check |
|---|---|---|
| 1 | `pgvector_table_schema_matches` | The `chunks` table exists with the exact columns and types from `contracts/chunks.json`: `id BIGINT`, `doc_id BIGINT`, `chunk_text TEXT`, `embedding VECTOR(384)`, `tokens INTEGER`, `created_at TIMESTAMPTZ`. |
| 2 | `chunks_count_matches_contract` | After ingesting the 5,000 docs, `COUNT(chunks)` is in `[30,000, 100,000]`. Not 5,000 (one chunk per doc = too coarse). Not 1M (too fine = loss of semantic context). |
| 3 | `embeddings_deterministic` | Embedding the SAME text TWICE produces the SAME vector, bit-for-bit. Vectors are L2-normalized (norm ≈ 1.0). |
| 4 | `retrieval_recall_at_5` | On the 50 golden questions, recall@5 ≥ 0.70: the correct doc appears in the top-5 for at least 35/50 questions. |
| 5 | `retrieval_mrr` | MRR ≥ 0.45. Measures ranking quality, not just presence — to catch "I find the right doc but at position 6". |
| 6 | `hallucination_gate_triggers` | On 10 out-of-domain questions (soccer, cooking, weather), `generate_answer()` must refuse to answer (return None or raise) at least 9 times out of 10. |

---

## The traps we see in senior code review

- **A chunker that splits memo codes in half.**
  Each document in the corpus contains a unique identifier of the form
  `RX-7K3M-2H8P`. Golden questions look for these codes. If your
  chunker does fixed-size 100 chars without a clean split, the code
  ends up straddling two chunks — and your embedder, seeing `RX-7K3M-2`
  then `H8P-2H8P`, no longer recognizes anything. Recall collapses.
  Fix: split by sentence first, then re-batch by size.

- **Non-deterministic embeddings in CI.**
  Three classic causes:
  1. Dropout active. `sentence-transformers` sets the model to
     `.eval()` by default — don't pull it out of that.
  2. GPU with non-deterministic cuDNN. In CI we're on CPU, so OK; but
     if you test on a local GPU, you may see `1e-6` deltas that break
     an `array_equal`.
  3. Shuffled inputs. `model.encode(texts)` must return vectors in the
     order of `texts`. If you shuffle, you desync.

- **Confusing cosine and L2 in pgvector.**
  `<=>` is cosine distance, `<->` is L2 distance, `<#>` is negative
  inner-product. On **L2-normalized** vectors, cosine and inner-product
  give the **same ranking** (and inner-product is faster). On
  non-normalized vectors, L2 favors short vectors — your ranking is
  wrong and you don't know why.
  **Rule**: `normalize_embeddings=True` everywhere + `<=>` or `<#>`
  (never `<->`).

- **Conflating recall and precision.**
  Recall@5 = "is the right doc in the top-5?". MRR = "how high is it in
  the top-5?". A pipeline with recall@5 = 0.95 but MRR = 0.20 is broken:
  it finds the right doc, but always at position 4. An assistant that
  reads only the top-1 would hallucinate 80% of the time. That's why
  the rubric demands both.

- **No hallucination gate.**
  Retrieval does NOT KNOW that it doesn't have the answer. If you
  naively grab the top-1 and feed it to the LLM, the LLM will invent a
  plausible answer even when the score is 0.05. Put a hard threshold
  (0.35 by default) on the top-1 score and **refuse to answer** below
  it. It's the cheapest, highest-ROI quality control in a RAG.

- **No ANN index.**
  Not required by the rubric (50k chunks ≈ 50 ms per query in a
  sequential scan). But in production, at 5M chunks, that's 5 seconds
  per query. `CREATE INDEX chunks_embedding_ivfflat_idx ON chunks USING
  ivfflat (embedding vector_cosine_ops) WITH (lists = 100);` or its
  HNSW equivalent. Document your choice in `docs/decisions.md`.

- **Truncate + insert vs upsert.**
  On 50k chunks, `TRUNCATE chunks; INSERT ...` is acceptable (~5s). In
  production on millions of chunks, that would be suicidal. The
  long-term best practice: an idempotent INSERT on a natural key (e.g.
  `UNIQUE (doc_id, content_hash)`) with `ON CONFLICT DO NOTHING`. Pick
  one or the other, document it.

---

## Going further

No reading is mandatory, but here's what shapes the rubric on the
state-of-the-art 2026 side:

- **Reis & Housley**, *Fundamentals of Data Engineering* (O'Reilly,
  2022) — ch. 8 "Queries, Modeling, Transformation", p. 247 ff. on
  batch transformations and coupling with the serving layer.
- **pgvector**: [official README](https://github.com/pgvector/pgvector)
  — read the operators section (`<=>`, `<->`, `<#>`) carefully, the
  `ivfflat` vs `hnsw` indexes, and the `lists` / `probes` parameters.
- **sentence-transformers**: [API docs](https://www.sbert.net/) —
  understand `normalize_embeddings`, `batch_size`, and why
  `all-MiniLM-L6-v2` is a reasonable default for English on CPU (384
  dims, 80 MB, ~5,000 sentences/s batched).
- **Kleppmann**, *Designing Data-Intensive Applications* (O'Reilly,
  2017) — ch. 3 "Storage and Retrieval", to understand why a vector
  index is not a B-tree.
- **Lewis et al.** (2020), *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks* — the foundational RAG paper. Short,
  worth reading for the philosophy, after you've coded.

---

## If you're stuck

The point is for you to struggle — this is a senior project, and that
struggle is what the investment covers.

1. Re-read the test error message — it almost always points at the
   cause.
2. Run the pipeline step by step: `python -m src.chunk`, then `python
   -m src.embed` (sanity check on 3 texts), then `python -m src.store`,
   then `python -m src.retrieve`.
3. Inspect the Postgres table: `psql -h localhost -U rag -d rag -c
   "SELECT COUNT(*), AVG(tokens) FROM chunks;"`.
4. If your embeddings are deterministic locally but not in CI, it's
   probably the model version. Verify that `EMBEDDING_MODEL` is set to
   `sentence-transformers/all-MiniLM-L6-v2`.
5. Open an issue on your fork with the `help-wanted` label — the
   IAmDataEng community hangs out there.

Good luck.
