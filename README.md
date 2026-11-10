# Srot (स्रोत)

Forensic source tracing for Hindi text. Given any Hindi text, Srot will find where each
sentence came from and infer the chain of transformations used to disguise it, with a
confidence score and evidence. It's built to stay robust when an adversarial attacker
agent stacks evasions on top of each other.

**Status: Phase 0 (audit and restructure).** The corpus is extracted and documented.
There is no detection code yet.

## Layout

```
data/raw/{topic}/*.docx          original corpus files, untouched
data/corpus/{topic}/{doc_id}.txt extracted text (UTF-8, NFC), one paragraph per line
data/corpus/manifest.jsonl       one row per document: ids, counts, duplicates, notes
data/resources/lemmas.tsv        merged hand-made lemma pairs    (NOT approved; unused)
data/resources/synonyms.tsv      merged hand-made synonym pairs  (NOT approved; unused)
data/resources/review_notes.tsv  hand-written reasons for flagged pairs
scripts/                         build_corpus.py, build_resources.py, corpus_report.py
src/srot/                        the package (empty for now)
legacy/                          the old prototype, unchanged (see legacy/README.md)
DECISIONS.md                     why things are the way they are
```

## Setup

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .

uv run python scripts/build_corpus.py      # regenerate data/corpus (deterministic)
uv run python scripts/build_resources.py   # regenerate data/resources/*.tsv
uv run python scripts/corpus_report.py     # print the corpus report
```

## Reviewing the lexicons

In `data/resources/lemmas.tsv` and `synonyms.tsv`, set `status` to `approved` or `rejected`.
- `review` rows were flagged as suspicious; the `reason` column says why.
- `pending` rows weren't flagged, but are still unapproved.

Re-running `build_resources.py` keeps your `approved`/`rejected` decisions.
