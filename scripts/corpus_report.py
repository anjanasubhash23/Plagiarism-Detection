"""Print a short report on the extracted corpus.

    uv run python scripts/corpus_report.py

Reads data/corpus/manifest.jsonl and the .txt files it points to.
"""

import itertools
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "manifest.jsonl"

SHINGLE_SIZE = 5  # characters
JACCARD_THRESHOLD = 0.25


def char_shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """All overlapping character n-grams, after collapsing whitespace.

    Character n-grams need no tokeniser and still match when words are split,
    joined or slightly misspelt, which suits a quick near-duplicate scan.
    """
    text = " ".join(text.split())
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """|A ∩ B| / |A ∪ B|: 1.0 for identical sets, 0.0 for nothing shared."""
    return len(a & b) / len(a | b) if a or b else 0.0


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    sources = [r for r in rows if r["duplicate_of"] is None]

    print("Corpus summary (duplicates excluded)\n")
    print(f"{'topic':<12}{'docs':>6}{'sentences':>11}{'words':>8}{'words/doc':>11}")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in sources:
        t = totals[r["topic"]]
        t[0] += 1
        t[1] += r["sentence_count"]
        t[2] += r["word_count"]
    for topic, (docs, sents, words) in sorted(totals.items()):
        print(f"{topic:<12}{docs:>6}{sents:>11}{words:>8}{words / docs:>11.0f}")
    docs, sents, words = (sum(t[i] for t in totals.values()) for i in range(3))
    print(f"{'TOTAL':<12}{docs:>6}{sents:>11}{words:>8}{words / docs:>11.0f}")

    duplicates = [r for r in rows if r["duplicate_of"]]
    print(f"\nDeclared exact duplicates: {len(duplicates)}")
    for r in duplicates:
        print(f"  {r['doc_id']} = {r['duplicate_of']}  ({r['raw_path']})")

    noted = [r for r in sources if r["notes"]]
    print(f"\nDocuments with notes: {len(noted)}")
    for r in noted:
        print(f"  {r['doc_id']:<15} {r['notes']}")

    shingles = {
        r["doc_id"]: char_shingles((ROOT / r["path"]).read_text(encoding="utf-8")) for r in sources
    }
    pairs = []
    for a, b in itertools.combinations(sorted(shingles), 2):
        score = jaccard(shingles[a], shingles[b])
        if score > JACCARD_THRESHOLD:
            pairs.append((score, a, b))
    print(
        f"\nNear-duplicate pairs (character {SHINGLE_SIZE}-gram Jaccard > {JACCARD_THRESHOLD}): "
        f"{len(pairs)} of {len(shingles) * (len(shingles) - 1) // 2} pairs"
    )
    for score, a, b in sorted(pairs, reverse=True):
        print(f"  {score:.3f}  {a} ~ {b}")

    all_scores = sorted(
        (jaccard(shingles[a], shingles[b]), a, b)
        for a, b in itertools.combinations(sorted(shingles), 2)
    )
    below = [p for p in all_scores if p[0] <= JACCARD_THRESHOLD]
    print("\nHighest-scoring pairs below the threshold (for calibration):")
    for score, a, b in below[-3:][::-1]:
        print(f"  {score:.3f}  {a} ~ {b}")


if __name__ == "__main__":
    main()
