"""Merge the prototype's hand-made lemma and synonym lists into two TSV files.

    uv run python scripts/build_resources.py

Reads   legacy/**/lemma*.txt, synonym*.txt, ...   (format "word : target", one per line)
        data/resources/review_notes.tsv            (hand-written flags with reasons)
Writes  data/resources/lemmas.tsv
        data/resources/synonyms.tsv

Columns: word, target, status, reason, sources
- status "review":  flagged as suspicious (see reason); a human must decide.
- status "pending": nothing suspicious found, but still not approved.
  A human changes status to "approved" or "rejected"; re-running this script keeps
  those decisions. NOTHING in src/ may use these files until rows are approved
  (DECISIONS.md).
- sources: every "file:line" the pair came from, so each row is traceable.
"""

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "resources"

LEMMA_FILES = [
    "legacy/lemma.txt",
    "legacy/lemma (2).txt",
    "legacy/Mahabharat/lemma (3).txt",
    "legacy/Nisarga/lemmuu.txt",
]
SYNONYM_FILES = [
    "legacy/synonym (2).txt",
    "legacy/synonym (4).txt",
    "legacy/Mahabharat/sy.txt",
    "legacy/Nisarga/synonymmmu.txt",
]

Pair = tuple[str, str]
HUMAN_STATUSES = {"approved", "rejected"}  # set by hand; preserved across re-runs


def clean(text: str) -> str:
    """NFC and collapse runs of whitespace (the files have stray double spaces)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_pairs(files: list[str]) -> tuple[dict[Pair, list[str]], int]:
    """Map each (word, target) pair to the list of 'file:line' it appears on.

    Pairs where word == target carry no information and are dropped (counted).
    """
    sources: dict[Pair, list[str]] = defaultdict(list)
    identical = 0
    for name in files:
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            word, target = (clean(part) for part in line.split(":", 1))
            if word == target:
                identical += 1
                continue
            sources[(word, target)].append(f"{name}:{number}")
    return sources, identical


def automatic_reasons(pairs: dict[Pair, list[str]], kind: str) -> dict[Pair, list[str]]:
    """Mechanical checks. They catch obvious problems, not wrong meanings."""
    targets_by_word: dict[str, set[str]] = defaultdict(set)
    for word, target in pairs:
        targets_by_word[word].add(target)

    reasons: dict[Pair, list[str]] = defaultdict(list)
    for word, target in pairs:
        if kind == "lemma":
            # A word has one lemma; several means at least one is wrong.
            if len(targets_by_word[word]) > 1:
                others = sorted(targets_by_word[word] - {target})
                reasons[(word, target)].append(f"conflicting lemmas: also {', '.join(others)}")
            # Inflection changes the end of a Hindi word, not the start.
            if word[0] != target[0]:
                reasons[(word, target)].append("lemma does not start like the word")
        if len(word.split()) != len(target.split()):
            reasons[(word, target)].append("phrase vs single word")
    return reasons


def read_review_notes() -> dict[tuple[str, str, str], str]:
    """(kind, word, target) -> reason, from the hand-written review file."""
    notes = {}
    with (OUT_DIR / "review_notes.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            notes[(row["kind"], clean(row["word"]), clean(row["target"]))] = row["reason"]
    return notes


def read_human_decisions(path: Path) -> dict[Pair, str]:
    """Statuses a human already set in an existing TSV, so re-running never loses them."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {
            (row["word"], row["target"]): row["status"]
            for row in csv.DictReader(f, delimiter="\t")
            if row["status"] in HUMAN_STATUSES
        }


def write_tsv(kind: str, files: list[str], notes: dict[tuple[str, str, str], str]) -> None:
    path = OUT_DIR / f"{kind}s.tsv"
    decisions = read_human_decisions(path)
    pairs, identical = read_pairs(files)
    reasons = automatic_reasons(pairs, kind)
    for (note_kind, word, target), reason in notes.items():
        if note_kind == kind:
            assert (word, target) in pairs, f"review note for unknown pair: {word} : {target}"
            reasons[(word, target)].append(reason)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "target", "status", "reason", "sources"])
        for (word, target), where in sorted(pairs.items()):
            why = "; ".join(reasons.get((word, target), []))
            status = decisions.get((word, target)) or ("review" if why else "pending")
            writer.writerow([word, target, status, why, " | ".join(where)])

    flagged = sum(1 for pair in pairs if reasons.get(pair))
    print(
        f"{path.name}: {len(pairs)} unique pairs, {flagged} flagged for review, "
        f"{identical} 'word : same word' lines dropped"
    )


def main() -> None:
    notes = read_review_notes()
    write_tsv("lemma", LEMMA_FILES, notes)
    write_tsv("synonym", SYNONYM_FILES, notes)


if __name__ == "__main__":
    main()
