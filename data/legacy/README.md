# Legacy data from the original prototype

These files were moved here unchanged (only renamed) from the original repo. They are
kept for reference and possible reuse. **Nothing in `src/` depends on them yet.**

| File | Original name | What it is | Caveats |
|---|---|---|---|
| `stopwords_and_demo_pair.py` | `stopwords.py` | Two stopword lists (`hi_stop`, 233 words; `hi_stop1`, 247 words) plus the one hardcoded demo pair (`Original_text`, `Plagarized_text`) | `hi_stop1` adds words that occur in the demo's suspect text (e.g. किंतु, प्रत्येक, नही), so it was tuned to that one example |
| `lexicons/lemma_v1.txt` | `lemma.txt` | `word : lemma`, 245 lines | Hand-made, small |
| `lexicons/lemma_v2.txt` | `lemma (2).txt` | `word : lemma`, 340 lines, used by the old `model.py` | Overlaps with v1, but v1 is not a subset of it |
| `lexicons/lemma_mahabharat.txt` | `Mahabharat/lemma (3).txt` | `word : lemma`, 177 lines | |
| `lexicons/lemma_nisarga.txt` | `Nisarga/lemmuu.txt` | `word : lemma`, 298 lines | |
| `lexicons/synonym_demo.txt` | `synonym (2).txt` | `variant : canonical`, 96 lines, loaded by the old `model.py` (but the result was never used) | Maps the demo pair's substitutions back to the original's words, and some entries are not synonyms (e.g. महसूस : सताती) |
| `lexicons/synonym_general.txt` | `synonym (4).txt` | `word : synonym`, 317 lines | Quality not yet reviewed |
| `lexicons/synonym_mahabharat.txt` | `Mahabharat/sy.txt` | `word : synonym`, 214 lines | Probably used to create paraphrased test docs |
| `lexicons/synonym_nisarga.txt` | `Nisarga/synonymmmu.txt` | `word : synonym`, 124 lines | Some entries are wrong (e.g. गुर्राना : भौंकना) |
| `pos/hindi_nltk.txt` | `hindipos.txt` | NLTK's Indian-languages Hindi POS corpus (540 sentences) | |
| `pos/hindi_nltk_plus2.txt` | `hinpos.txt` | The same plus 2 hand-tagged sentences | |
| `pos/tagged_mahabharat.txt` | `Mahabharat/only_my_docPOS.txt` | POS-tagged, preprocessed Mahabharat text | |
| `mahabharat_old_scores.docx` | `Mahabharat/ACCU_PER.docx` | Scores the old system gave DOC11–DOC20 | The suspect documents behind these scores are not in the repo, and the old scoring was broken, so these numbers can't be reproduced or relied on |
