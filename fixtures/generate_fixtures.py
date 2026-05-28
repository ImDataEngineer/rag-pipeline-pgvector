"""Deterministic synthetic corpus generator for the RAG pipeline project.

Run from the project root:

    python -m fixtures.generate_fixtures

Outputs three files under fixtures/:

    documents.jsonl       — 5 000 support-ticket articles
                             ({"doc_id": int, "title": str, "body": str, "category": str})
    golden_eval.json      — 50 evaluation questions with ground-truth doc_id
                             ({"question": str, "expected_doc_id": int, "expected_passage": str})
    out_of_domain.json    — 10 off-topic questions (football, cooking, weather)
                             that the hallucination_gate MUST refuse

Why synthetic and not a real public corpus?
- Reproducibility: a real corpus drifts over time; CI must be byte-stable.
- Right-sized: 5 000 docs is enough to make retrieval non-trivial (millions
  of candidate chunks would not change the engineering, only the cost).
- Identifiable signal: each document carries a UNIQUE 12-character "memo
  code" (e.g. `RX-7K3M-2H8P`) in its body. The golden questions ask about a
  specific memo code, so a correct retrieval pipeline reliably returns the
  right doc — without us having to ship a real test set.

The fictional domain is "Lumora Stock" — a B2B SaaS for inventory management.
All article styles (FAQ, troubleshooting, how-to, changelog) are mixed in
to exercise the chunker's robustness to varied formats.

DETERMINISM CONTRACT
- Single `random.Random(SEED)` instance threaded through every choice.
- No time-based values, no platform-dependent encoding.
- JSON written with `sort_keys=False` and `ensure_ascii=False`, but the
  insertion order of every dict is fixed in source — so output is identical
  on macOS, Linux, and the Codespaces image.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

SEED = 42
N_DOCUMENTS = 5_000
N_GOLDEN_QUESTIONS = 50
N_OUT_OF_DOMAIN = 10

FIXTURES_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Domain vocabulary — fictional SaaS "Lumora Stock"
# ---------------------------------------------------------------------------

PRODUCTS = [
    "Lumora Inventory", "Lumora Pulse", "Lumora Forecast",
    "Lumora Sync", "Lumora Connect", "Lumora Reports",
]

MODULES = [
    "stock count", "purchase order", "supplier sync", "barcode scan",
    "reorder rule", "shelf-life alert", "cycle count", "transfer order",
    "kit assembly", "consignment", "return-to-vendor", "stock adjustment",
]

ERRORS = [
    "E-4001 supplier API timeout",
    "E-4002 SKU mapping conflict",
    "E-4003 unit-of-measure mismatch",
    "E-4004 negative stock not allowed",
    "E-4005 warehouse zone full",
    "E-4006 reorder threshold misconfigured",
    "E-4007 barcode format unsupported",
    "E-4008 shelf-life parser rejected the batch",
    "E-4009 transfer order missing destination",
    "E-4010 cycle-count session expired",
]

ROLES = ["warehouse manager", "operations lead", "buyer", "auditor", "store manager"]

STYLES = ["faq", "troubleshooting", "howto", "changelog"]


# ---------------------------------------------------------------------------
# Memo codes — unique per document, the retrieval signal
# ---------------------------------------------------------------------------

_MEMO_ALPHABET = string.ascii_uppercase + string.digits


def _make_memo_code(rng: random.Random) -> str:
    """Generates a unique 12-char memo code like `RX-7K3M-2H8P`.

    Uniqueness across the whole corpus is enforced by the caller by drawing
    without replacement from a precomputed set. Format is intentionally weird
    so the embedder cannot match it via fuzzy semantic similarity alone — the
    chunker must actually carry the literal token through to the index.
    """
    segs = ["".join(rng.choice(_MEMO_ALPHABET) for _ in range(2 + i % 2 * 2))
            for i in range(3)]
    return f"{segs[0]}-{segs[1]}-{segs[2]}"


# ---------------------------------------------------------------------------
# Document generators — one function per style.
# Each returns (title, body, category, memo_code).
# Body MUST contain the memo code as a verbatim token.
# ---------------------------------------------------------------------------


def _gen_faq(rng: random.Random, memo: str) -> tuple[str, str, str]:
    product = rng.choice(PRODUCTS)
    module = rng.choice(MODULES)
    role = rng.choice(ROLES)
    title = f"FAQ — {product}: how does the {module} work for a {role}?"
    body = (
        f"Reference: {memo}.\n"
        f"Question: A {role} needs to understand how the {module} works in {product}.\n"
        f"Answer: In {product}, the {module} is triggered whenever a SKU crosses "
        f"its configured threshold. Permissions follow the workspace role matrix. "
        f"For audit purposes every transition is logged with the memo code {memo}. "
        f"See the related how-to article in the {rng.choice(MODULES)} section."
    )
    return title, body, "faq"


def _gen_troubleshooting(rng: random.Random, memo: str) -> tuple[str, str, str]:
    err = rng.choice(ERRORS)
    product = rng.choice(PRODUCTS)
    title = f"Troubleshooting — {err} in {product}"
    body = (
        f"Symptom: users of {product} hit {err} when running a {rng.choice(MODULES)}.\n"
        f"Diagnosis: the most common cause is a misaligned configuration between the "
        f"workspace and the connected supplier system. Check the audit memo {memo} in "
        f"the incident log to confirm.\n"
        f"Fix: re-run the wizard under Settings > Integrations, then retry the failed "
        f"operation. If the error persists, escalate with memo code {memo}."
    )
    return title, body, "troubleshooting"


def _gen_howto(rng: random.Random, memo: str) -> tuple[str, str, str]:
    module = rng.choice(MODULES)
    product = rng.choice(PRODUCTS)
    title = f"How to configure {module} in {product}"
    body = (
        f"This how-to walks through {module} setup in {product}.\n"
        f"Step 1: open the workspace settings panel.\n"
        f"Step 2: select the {module} tab and click 'New rule'.\n"
        f"Step 3: enter a name, choose the scope, save. Internally the rule is "
        f"tagged with memo code {memo} for traceability.\n"
        f"Step 4: validate by running a dry-run from the {rng.choice(MODULES)} "
        f"console. Successful runs include the same memo code in their audit trail."
    )
    return title, body, "howto"


def _gen_changelog(rng: random.Random, memo: str) -> tuple[str, str, str]:
    product = rng.choice(PRODUCTS)
    version = f"{rng.randint(2, 6)}.{rng.randint(0, 12)}.{rng.randint(0, 30)}"
    title = f"Changelog — {product} {version}"
    body = (
        f"{product} {version} — release notes.\n"
        f"Added: a new {rng.choice(MODULES)} workflow with explicit memo tagging.\n"
        f"Changed: the {rng.choice(MODULES)} engine now batches updates of up to 500 "
        f"items per request.\n"
        f"Fixed: edge case in {rng.choice(ERRORS)} when the supplier API returned a "
        f"redirect. Internal tracking memo: {memo}."
    )
    return title, body, "changelog"


_GENERATORS = {
    "faq": _gen_faq,
    "troubleshooting": _gen_troubleshooting,
    "howto": _gen_howto,
    "changelog": _gen_changelog,
}


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


def _draw_unique_memos(rng: random.Random, n: int) -> list[str]:
    """Draw `n` unique memo codes."""
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        code = _make_memo_code(rng)
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _gen_documents(rng: random.Random) -> list[dict]:
    memos = _draw_unique_memos(rng, N_DOCUMENTS)
    docs: list[dict] = []
    for doc_id, memo in enumerate(memos, start=1):
        style = STYLES[doc_id % len(STYLES)]  # deterministic style rotation
        title, body, category = _GENERATORS[style](rng, memo)
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "body": body,
            "category": category,
            "memo_code": memo,
        })
    return docs


# ---------------------------------------------------------------------------
# Golden evaluation set
# ---------------------------------------------------------------------------


_QUESTION_TEMPLATES = [
    "What does the audit memo {memo} refer to?",
    "I see memo code {memo} in our logs — which article documents it?",
    "Which support article mentions the reference {memo}?",
    "Can you point me to the documentation tagged with memo {memo}?",
    "We have an incident with memo {memo}. Where is it documented?",
]


def _gen_golden(rng: random.Random, docs: list[dict]) -> list[dict]:
    """Pick N_GOLDEN_QUESTIONS docs spread across the corpus and craft a Q for each.

    The "expected_passage" is a substring of the body — a learner whose chunker
    drops the memo code into the wrong chunk will fail the recall check.
    """
    indices = rng.sample(range(len(docs)), N_GOLDEN_QUESTIONS)
    questions: list[dict] = []
    for idx in indices:
        doc = docs[idx]
        template = rng.choice(_QUESTION_TEMPLATES)
        question = template.format(memo=doc["memo_code"])
        # The expected passage is the sentence that contains the memo code.
        # If the chunker preserves it, retrieval will work.
        passage = next(
            (s.strip() for s in doc["body"].split("\n") if doc["memo_code"] in s),
            doc["body"][:200],
        )
        questions.append({
            "question": question,
            "expected_doc_id": doc["doc_id"],
            "expected_memo": doc["memo_code"],
            "expected_passage": passage,
        })
    return questions


# ---------------------------------------------------------------------------
# Out-of-domain questions (hallucination_gate fodder)
# ---------------------------------------------------------------------------

OUT_OF_DOMAIN_QUESTIONS = [
    "Who won the FIFA World Cup in 2018?",
    "What is the recipe for a classic French ratatouille?",
    "How do I bake sourdough bread at home?",
    "Will it rain in Paris tomorrow?",
    "What is the offside rule in football?",
    "Recommend a good white wine for grilled fish.",
    "How long should I marinate chicken before grilling?",
    "What is the capital of Mongolia?",
    "Explain the rules of cricket in two sentences.",
    "What time does the next solar eclipse start in Europe?",
]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def _write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate() -> tuple[Path, Path, Path]:
    rng = random.Random(SEED)

    documents = _gen_documents(rng)
    golden = _gen_golden(rng, documents)
    ood = [{"question": q} for q in OUT_OF_DOMAIN_QUESTIONS[:N_OUT_OF_DOMAIN]]

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    docs_path = FIXTURES_DIR / "documents.jsonl"
    golden_path = FIXTURES_DIR / "golden_eval.json"
    ood_path = FIXTURES_DIR / "out_of_domain.json"

    _write_jsonl(docs_path, documents)
    _write_json(golden_path, golden)
    _write_json(ood_path, ood)
    return docs_path, golden_path, ood_path


def main() -> None:
    docs_path, golden_path, ood_path = generate()
    print(f"wrote {docs_path}     ({N_DOCUMENTS} documents)")
    print(f"wrote {golden_path}   ({N_GOLDEN_QUESTIONS} golden questions)")
    print(f"wrote {ood_path}      ({N_OUT_OF_DOMAIN} out-of-domain questions)")


if __name__ == "__main__":
    main()
