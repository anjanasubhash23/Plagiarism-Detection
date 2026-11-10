# Design decisions

Each entry says what we chose, why, and what we gave up. Entries are added phase by phase.

## Phase 1: Foundation

### D1. Rebuild rather than patch
The old `model.py` could not be fixed in place. It fitted TF-IDF separately on each text,
so the two vectors had different vocabularies and the cosine compared unrelated numbers.
On top of that, the two texts went through different preprocessing (different stopword
lists, POS files and dropped tags), and those choices were tuned to the single demo pair.
We start from a clean package and keep the old code reachable through the git tag
`v0-original`.

### D2. Corpus files are moved, never modified or deleted
The 65 `.docx` files are the project's most valuable asset. They were moved with `git mv`,
so history follows them, into `data/corpus/<domain>/` (domain folders lowercased, file names
unchanged for provenance). `data/corpus/SHA256SUMS` records their checksums, and
`tests/test_corpus.py` fails if any file changes, disappears or appears.

### D3. Duplicates are declared, not deleted
`mahabharat/Copy of doc_7.docx` is an exact copy of `doc_7.docx`. Because of D2 it stays on
disk, but `KNOWN_DUPLICATES` in `corpus.py` makes the loader skip it by default. Otherwise
the benchmark would contain a "plagiarised" pair that is really just one file stored twice.
A test checks that the declared pair is identical and that no other exact duplicates exist.

### D4. Read .docx with the standard library
A `.docx` is a zip file containing XML. Extracting paragraph text takes about 20 lines with
`zipfile` + `xml.etree`, so we don't add `python-docx`. The loader returns text exactly as
written, without normalisation, so cleaning stays a separate, testable step (Phase 2).

### D5. Prototype data goes in `data/legacy/`, documented
Lemma lists, synonym lists, stopwords, POS training files and old scores were renamed to
descriptive names and documented in `data/legacy/README.md`, including the known quality
problems. No new code depends on them; later phases will decide, using the benchmark, whether
any of them earn a place.

### D6. Removed: old app, `plag.py`, `requirements.txt`, `__pycache__`
- `plag.py` duplicated `model.py`.
- `requirements.txt` pinned about 211 packages when about 6 were used.
- `__pycache__` holds generated files that should never be committed.
- The Flask app (`app.py`, `templates/`, `static/`) will be replaced in Phase 5 once we've
  chosen between Gradio and Flask.

All of it is recoverable with `git show v0-original:<path>`.

### D7. Tooling: uv, src layout, Python 3.12 locally, no runtime dependencies yet
- **uv with `pyproject.toml`** gives one tool for environments, locking and running.
- **The src layout** (`src/hindi_plag/`) means tests import the installed package, not
  whatever files happen to be in the working directory.
- **Python:** `requires-python >= 3.11` is the support floor, and `.python-version` pins 3.12
  for development because ML libraries we may need later reliably ship wheels for it.
- **Dependencies:** the only one so far is `pytest`, in the dev group. Each later
  dependency must be used by code in the same phase.

### D8. Locating the corpus
`default_corpus_dir()` resolves `data/corpus` relative to the source file. This works for the
editable install that `uv sync` creates and for a Hugging Face Space, which runs from a repo
checkout. It would not work from a built wheel, but we don't plan to publish one.
