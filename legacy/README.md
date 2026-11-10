# legacy/: the original prototype (kept for reference only)

Everything here was moved unchanged from the original repo (tag `v0-original`).
Each file keeps its original relative path: `legacy/X` is what used to be at `X`.
**Nothing in `src/srot/` imports or reuses any of it.**

## What the old system did

A Flask app (`app.py`, `templates/`, `static/`) took two texts from a form and showed a
"plagiarism %". The score came from `model.py`:

1. Tokenise both texts with NLTK.
2. Remove stopwords: `hi_stop` for the original, `hi_stop1` for the suspect.
3. Replace words using a hand-made lemma list (`lemma (2).txt`).
4. Train a TnT POS tagger from scratch, twice per request: on NLTK's `hindi.txt` for the
   original, and on `hindipos.txt` for the suspect.
5. Drop proper nouns (NNP) from the original but common nouns (NN) from the suspect.
6. Look up synonyms (`synonym (2).txt`). The result (`synonym_data`) was never used.
7. Make word trigrams and fit a TF-IDF vectoriser separately on each text.
8. Take the raw non-zero weights from each matrix, zero-pad the shorter list, and return
   the cosine × 100.

## The scoring bug

Step 7–8 is why the score is meaningless:
- **The vectors aren't aligned.** Because TF-IDF was fitted separately, the two vectors have
  different vocabularies. Position *k* in one list and position *k* in the other refer to
  unrelated trigrams.
- **The order is arbitrary.** The values were read in sparse-matrix (`todok()`) order, which
  has nothing to do with the vocabulary.
- **The padding is meaningless.** Zero-padding the shorter list makes the arrays the same
  length but doesn't put them on a shared axis.

The cosine of two such lists measures nothing about shared content. Two texts with no
words in common can score high.

On top of that:
- The two texts went through different pipelines (steps 2, 4 and 5).
- `hi_stop1` and `synonym (2).txt` contain entries that exactly match the one hard-coded
  demo pair in `stopwords.py`, so the pipeline was tuned to that single example.
- `plag.py` is an older copy of the same pipeline that runs on that demo pair at import time.

## Other problems
- `app.py` reads form fields by position, and crashes on GET because `result` is undefined.
- The form posts to a hard-coded `onrender.com` URL. `static/app.js` wires up sign-in/sign-up
  buttons left over from a recycled login template.
- `requirements.txt` pins about 211 packages; about 6 were used.

## File map

| Path | What it is |
|---|---|
| `app.py`, `templates/`, `static/` | Flask UI |
| `model.py` | Scorer used by the app |
| `plag.py` | Older duplicate of `model.py` |
| `stopwords.py` | `hi_stop`, `hi_stop1`, and the demo pair `Original_text` / `Plagarized_text` |
| `requirements.txt` | Pinned environment |
| `hindipos.txt` | NLTK Hindi POS corpus (540 sentences) |
| `hinpos.txt` | The same plus 2 hand-tagged sentences |
| `Mahabharat/only_my_docPOS.txt` | POS-tagged, preprocessed Mahabharat text |
| `Mahabharat/ACCU_PER.docx` | Old scores for "DOC11"–"DOC20". The suspect docs aren't in the repo, and the scorer was broken, so these numbers can't be reproduced or relied on |
| `lemma.txt`, `lemma (2).txt`, `Mahabharat/lemma (3).txt`, `Nisarga/lemmuu.txt` | Hand-made lemma lists. Merged into `data/resources/lemmas.tsv` |
| `synonym (2).txt`, `synonym (4).txt`, `Mahabharat/sy.txt`, `Nisarga/synonymmmu.txt` | Hand-made synonym lists. Merged into `data/resources/synonyms.tsv` |
