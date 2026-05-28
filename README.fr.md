# Un pipeline RAG production-grade, mesuré — `transformation.rag-pipeline-pgvector`

> **Niveau** : senior · **Durée estimée** : ~12 h · **Tarif** : 49 €
> **Axe framework** : `transformation` (sous-compétences : `data_modeling`,
> `incremental_transforms`, `performance_tuning`)
> **Axes secondaires couverts** : `storage` (pgvector + index ANN),
> `software_engineering_dataops` (déterminisme, évaluation rigoureuse).

**Un pipeline RAG en production, c'est 80 % de DATA et 20 % de LLM. Ce projet
teste les 80 %.**

Si tu veux apprendre à "appeler GPT-4 avec un contexte", tu n'es pas au bon
endroit. Si tu veux apprendre à construire le pipeline qui, le jour où ton
LLM hallucine, te permet de prouver que c'est la faute du LLM et pas de ta
data — bienvenue.

La rubric ne juge **pas** la qualité des réponses générées. Elle juge la
qualité du pipeline : déterminisme, reproductibilité, observabilité,
évaluation rigoureuse. C'est ce qu'un recruteur tech senior teste en
interview RAG en 2026 — et c'est ce que la plupart des candidats foirent.

---

## Le contexte

Lumora Stock est une SaaS B2B de gestion de stock. Leur équipe support a
accumulé **5 000 articles** (FAQ, troubleshooting, how-to, changelogs) sur
quatre ans. L'équipe perd 4 h/jour à rechercher manuellement des réponses
déjà documentées.

Ton job : construire le pipeline qui ingère ces articles, les chunke
proprement, les embedde de façon reproductible, les stocke dans Postgres +
pgvector, répond à une question avec un retrieval **mesurable** (recall@5,
MRR), et **refuse de répondre** quand le retrieval est trop faible — pour
ne PAS halluciner sur les questions hors-domaine.

Stack :
- **Postgres 16 + pgvector** (`pgvector/pgvector:pg16`)
- **Python 3.11** + sqlalchemy + psycopg2 + numpy
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384 dims, local CPU)
- **pytest** pour la rubric
- *(optionnel)* **Ollama** + `qwen2.5:0.5b` si tu veux brancher une vraie
  génération — la rubric n'en a PAS besoin

Pas d'OpenAI, pas d'Anthropic, pas de clé API. Tout tourne en local, tout
est déterministe, tout est rejouable en CI.

---

## Ce que tu vas livrer

| Livrable | Où |
|---|---|
| Le découpage en chunks | `src/chunk.py` (`split_documents()`) |
| L'embedder | `src/embed.py` (`embed_chunks()`) |
| L'upsert pgvector | `src/store.py` (`init_schema()`, `upsert_to_pgvector()`) |
| La recherche top-k | `src/retrieve.py` (`top_k_search()`) |
| Le hallucination gate | `src/answer.py` (`generate_answer()`) |
| Une ADR (Architecture Decision Record) | `docs/decisions.md` — chunking choisi, seuil de confiance, index ANN si présent |

`src/evaluate.py` **est fourni en clair** (helpers `recall_at_k`, `MRR`).
Tu n'as pas à le réécrire — la rubric senior porte sur le pipeline, pas
sur la formule des métriques.

---

## Comment commencer

En Codespaces (ouverture en un clic depuis l'app IAmDataEng), le
post-create installe les dépendances, démarre Postgres+pgvector, pré-télécharge
le modèle sentence-transformers, et génère les fixtures. Donc tu n'as qu'à
coder.

En local :

```bash
# 1. Démarrer Postgres + pgvector
docker compose -f .devcontainer/docker-compose.yml up -d postgres

# 2. Installer les dépendances (CPU-only torch pour rester léger)
pip install -r requirements.txt

# 3. Générer les fixtures (déterministe — seed=42, ~5000 docs)
python -m fixtures.generate_fixtures

# 4. Implémenter src/*.py puis lancer le pipeline
python -m src.store

# 5. Lancer la rubric
pytest tests/ -v
```

Quand les 6 checks passent en local, **commit + push** sur ton fork. La CI
GitHub Actions rejoue la même rubric (avec un Postgres+pgvector éphémère
comme service container).

---

## Les 6 checks de la rubric

Définis dans `tests/test_evaluate.py`. Tous **déterministes**. Tous lisent
des **artefacts** (la table Postgres, les fonctions Python du learner) — pas
du style de code.

| # | Check | Ce qu'on vérifie |
|---|---|---|
| 1 | `pgvector_table_schema_matches` | La table `chunks` existe avec les colonnes et types exacts de `contracts/chunks.json` : `id BIGINT`, `doc_id BIGINT`, `chunk_text TEXT`, `embedding VECTOR(384)`, `tokens INTEGER`, `created_at TIMESTAMPTZ`. |
| 2 | `chunks_count_matches_contract` | Après ingestion des 5 000 docs, `COUNT(chunks)` est dans `[30 000, 100 000]`. Pas 5 000 (un chunk par doc = chunking trop grossier). Pas 1 M (chunking trop fin = perte de contexte sémantique). |
| 3 | `embeddings_deterministic` | Embedder DEUX FOIS le même texte produit le MÊME vecteur, bit-pour-bit. Vecteurs L2-normalisés (norme ≈ 1.0). |
| 4 | `retrieval_recall_at_5` | Sur les 50 questions golden, recall@5 ≥ 0.70 : le bon doc apparaît dans le top-5 pour au moins 35/50 questions. |
| 5 | `retrieval_mrr` | MRR ≥ 0.45. Mesure la qualité du ranking, pas juste la présence — pour traquer le cas « je trouve le bon doc mais en 6ᵉ position ». |
| 6 | `hallucination_gate_triggers` | Sur 10 questions hors-domaine (foot, cuisine, météo), `generate_answer()` doit refuser de répondre (return None ou raise) au moins 9 fois sur 10. |

---

## Les pièges qu'on voit en revue de code senior

- **Chunker qui coupe les codes mémo en deux.**
  Chaque document du corpus contient un identifiant unique de la forme
  `RX-7K3M-2H8P`. Les questions golden cherchent ces codes. Si ton chunker
  fait du fixed-size 100 chars sans découpe propre, le code se retrouve à
  cheval sur deux chunks — et ton embedder, qui voit `RX-7K3M-2` puis
  `H8P-2H8P`, ne reconnaît plus rien. Recall s'effondre. Solution : split
  par phrases d'abord, puis re-batch par taille.

- **Embeddings non-déterministes en CI.**
  Trois causes classiques :
  1. Dropout activé. `sentence-transformers` met le modèle en `.eval()` par
     défaut — ne le sors pas de là.
  2. GPU avec cuDNN non-déterministe. En CI on est en CPU, donc OK ;
     mais si tu testes en local sur GPU, tu peux voir des deltas de l'ordre
     de `1e-6` qui font foirer un `array_equal`.
  3. Shuffle des inputs. `model.encode(texts)` doit retourner les vecteurs
     dans l'ordre des `texts`. Si tu shuffles, tu désynchronises.

- **Confondre cosine et L2 dans pgvector.**
  `<=>` est cosine distance, `<->` est L2 distance, `<#>` est negative
  inner-product. Sur des vecteurs **L2-normalisés**, cosine et inner-product
  donnent le **même ranking** (et inner-product est plus rapide).
  Sur des vecteurs non-normalisés, L2 favorise les vecteurs courts — ton
  ranking est faux et tu ne comprends pas pourquoi.
  **Règle** : `normalize_embeddings=True` partout + `<=>` ou `<#>` (jamais
  `<->`).

- **Ignorer la différence entre recall et precision.**
  Recall@5 = « le bon doc est-il dans le top-5 ? ». MRR = « à quel point est-il
  haut dans le top-5 ? ». Un pipeline avec recall@5 = 0.95 mais MRR = 0.20
  est cassé : il trouve le bon doc, mais toujours en 4ᵉ position. Un
  assistant qui ne lit que le top-1 hallucinerait 80 % du temps. La rubric
  exige les deux pour cette raison.

- **Pas de hallucination gate.**
  Le retrieval ne SAIT PAS qu'il n'a pas la réponse. Si tu prends naïvement
  le top-1 et tu le passes au LLM, le LLM inventera une réponse plausible
  même quand le score est de 0.05. Mets un seuil dur (0.35 par défaut)
  sur le score du top-1 et **refuse de répondre** sous ce seuil.
  C'est le contrôle de qualité le moins cher et le plus rentable d'un RAG.

- **Pas d'index ANN.**
  Pas obligatoire pour la rubric (50k chunks ≈ 50ms par requête en
  sequential scan). Mais en prod, à 5 M chunks, c'est 5 secondes par
  requête. `CREATE INDEX chunks_embedding_ivfflat_idx ON chunks USING
  ivfflat (embedding vector_cosine_ops) WITH (lists = 100);` ou son
  équivalent HNSW. Documente ton choix dans `docs/decisions.md`.

- **Truncate + insert vs upsert.**
  Sur 50k chunks, `TRUNCATE chunks; INSERT ...` est acceptable (~5s).
  En prod sur des millions de chunks, ce serait suicidaire. La bonne
  pratique long-terme : un INSERT idempotent sur une clé naturelle
  (par ex. `UNIQUE (doc_id, content_hash)`) avec `ON CONFLICT DO NOTHING`.
  Choisis l'un ou l'autre, documente.

---

## Pour aller plus loin

Aucune lecture obligatoire, mais voici ce qui structure la rubric côté
état de l'art 2026 :

- **Reis & Housley**, *Fundamentals of Data Engineering* (O'Reilly, 2022) —
  chap. 8 « Queries, Modeling, Transformation », p. 247 sq. sur les
  transformations batch et le couplage avec le serving layer.
- **pgvector** : [README officiel](https://github.com/pgvector/pgvector)
  — lis attentivement la section sur les opérateurs (`<=>`, `<->`, `<#>`),
  les index `ivfflat` vs `hnsw`, et les paramètres `lists` et `probes`.
- **sentence-transformers** : [docs API](https://www.sbert.net/) —
  comprends `normalize_embeddings`, `batch_size`, et pourquoi
  `all-MiniLM-L6-v2` est le défaut raisonnable pour de l'anglais en CPU
  (384 dims, 80 MB, ~5 000 sentences/s en batched).
- **Kleppmann**, *Designing Data-Intensive Applications* (O'Reilly, 2017)
  — chap. 3 « Storage and Retrieval », pour comprendre pourquoi un index
  vectoriel n'est pas un B-tree.
- **Lewis et al.** (2020), *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks* — le papier fondateur du RAG. Court,
  utile pour comprendre la philosophie, à lire après avoir codé.

---

## Si tu es bloqué

L'objectif est que tu galères — c'est un projet senior et c'est ce que
recouvre l'investissement.

1. Relis le message d'erreur du test — il pointe presque toujours la cause.
2. Lance le pipeline étape par étape :
   `python -m src.chunk`, puis `python -m src.embed` (sanity check sur 3
   textes), puis `python -m src.store`, puis `python -m src.retrieve`.
3. Inspecte la table Postgres :
   `psql -h localhost -U rag -d rag -c "SELECT COUNT(*), AVG(tokens) FROM chunks;"`.
4. Si tes embeddings sont déterministes en local mais pas en CI, c'est
   probablement la version du modèle. Vérifie que `EMBEDDING_MODEL` est
   bien `sentence-transformers/all-MiniLM-L6-v2`.
5. Ouvre une issue dans ton fork avec le label `help-wanted` — la
   communauté IAmDataEng y passe.

Bonne route.
