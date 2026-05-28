#!/usr/bin/env bash
# IAmDataEng — first-boot bootstrap for `transformation.rag-pipeline-pgvector`.
#
# 1. Install Python deps
# 2. Start Postgres + pgvector via docker compose
# 3. Pre-download the sentence-transformers model (avoid CI cold-start timeout
#    and keep the learner's first `python -m src.embed` snappy)
# 4. Generate the deterministic synthetic corpus (5k docs, 50 golden Qs,
#    10 out-of-domain Qs)
#
# Re-runnable: every step is idempotent. `docker compose up -d` is a no-op
# when the container is already running, fixture regen is byte-stable thanks
# to seed=42, and the model download is cached in HF_HOME.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[2/4] Starting Postgres + pgvector..."
docker compose -f .devcontainer/docker-compose.yml up -d postgres

echo "[2.1/4] Waiting for Postgres to be healthy..."
for i in $(seq 1 60); do
  if docker exec rag-postgres pg_isready -U rag -d rag >/dev/null 2>&1; then
    echo "  Postgres ready (after ${i}s)"
    break
  fi
  if [ "$i" = "60" ]; then
    echo "  ERROR: Postgres did not become healthy within 60s."
    docker compose -f .devcontainer/docker-compose.yml logs --tail=50 postgres
    exit 1
  fi
  sleep 1
done

echo "[2.2/4] Ensuring pgvector extension is loaded..."
docker exec rag-postgres psql -U rag -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo "[3/4] Pre-downloading the sentence-transformers model (one-shot, ~80MB)..."
# Downloading at post-create avoids a cold cache the first time the learner
# runs the embedder. The model lands in $HF_HOME, which we pinned in
# devcontainer.json so it survives Codespaces rebuilds.
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
v = m.encode(['warmup'], normalize_embeddings=True)
print(f'  model warm — vector dim = {v.shape[1]}')
"

echo "[4/4] Generating deterministic fixtures (seed=42, ~5000 docs)..."
python -m fixtures.generate_fixtures

echo
echo "Setup done. Next steps:"
echo "  1. Read README.fr.md"
echo "  2. Implement src/chunk.py, src/embed.py, src/store.py, src/retrieve.py, src/answer.py"
echo "  3. Run the pipeline end-to-end:"
echo "       python -m src.chunk    # split fixtures/documents.jsonl into chunks"
echo "       python -m src.embed    # encode chunks → 384-dim vectors"
echo "       python -m src.store    # upsert into Postgres + pgvector"
echo "       python -m src.retrieve # demo top-k search"
echo "  4. Run the rubric: pytest tests/ -v"
