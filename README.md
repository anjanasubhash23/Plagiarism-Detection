# Hindi Plagiarism Detection

Layered plagiarism detection for Hindi text, with an evaluation benchmark.
This is a rebuild of an earlier Flask prototype (kept in git history under the tag `v0-original`).

**Status:** Phase 1 (foundation). The corpus loads and is integrity-checked, but there is no detector yet.

## Layout

```
data/corpus/<domain>/*.docx   65 Hindi source documents in 4 domains (byte-for-byte originals)
data/corpus/SHA256SUMS        checksums; a test fails if any corpus file changes
data/legacy/                  word lists, POS data and scores from the prototype (see its README)
src/hindi_plag/               the package
tests/                        pytest suite
DECISIONS.md                  why things are the way they are
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync          # creates .venv with Python 3.12 and dev tools
uv run pytest    # run the tests
```

```python
from hindi_plag.corpus import load_corpus

docs = load_corpus()          # 64 documents (the one known duplicate is skipped)
print(docs[0].doc_id, docs[0].text[:80])
```
