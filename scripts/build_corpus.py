"""Extract the raw .docx corpus into plain-text files plus a manifest.

    uv run python scripts/build_corpus.py

Reads   data/raw/{topic}/*.docx           (originals, never modified)
Writes  data/corpus/{topic}/{doc_id}.txt  (UTF-8, NFC, one paragraph per line)
        data/corpus/manifest.jsonl        (one JSON object per document)

The script is deterministic: running it twice gives byte-identical output.
See DECISIONS.md for why each rule below exists.
"""

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CORPUS_DIR = ROOT / "data" / "corpus"

# Exact copies: kept in data/raw (never delete originals) and listed in the
# manifest, but no .txt is written, so nothing downstream treats them as a source.
DUPLICATES = {"mahabharat/Copy of doc_7.docx": "mahabharat-07"}

# The prototype's lemma lists, POS-tagged text ("राम_NNP ...") and synonym lists were
# pasted into these files after the article. Only the first paragraph is the article.
ANNOTATED = {
    "paryatan/DOC3.docx",
    "paryatan/DOC4.docx",
    "paryatan/DOC5.docx",
    "paryatan/DOC6.docx",
    "paryatan/DOC10.docx",
}
POS_TAG = re.compile(r"_\s*[A-Z]{2,}")  # "राम_NNP" or "नगर _NN"

# Hand-written notes about known relationships between documents.
NOTES = {
    "nisarga-05": "Overlaps heavily with nisarga-06 (shared passage).",
    "nisarga-06": "Overlaps heavily with nisarga-05 (shared passage).",
    "ramayan-05": "Shares a passage with ramayan-08.",
    "ramayan-08": "Shares a passage with ramayan-05.",
}

# A "word" is a run of Devanagari letters/signs (minus the dandas), Latin letters or
# digits. Python's \w is not usable: it does not count vowel signs (matras) as word
# characters, so \w+ would split किताबें into क, त, ब.
WORD = re.compile(r"[ऀ-ॣ०-ॿ‌‍A-Za-z0-9]+")
SENTENCE_END = re.compile(r"[।?!\n]")

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx_paragraphs(path: Path) -> list[str]:
    """Non-empty paragraphs of a .docx (a zip whose text is in word/document.xml)."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.iter(f"{W_NS}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{W_NS}t":
                parts.append(node.text or "")
            elif node.tag in (f"{W_NS}tab", f"{W_NS}br"):
                parts.append(" ")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def make_doc_id(topic: str, filename: str) -> str:
    """'Doc7.docx', 'doc_7.docx', 'DOC7.docx' -> '<topic>-07'."""
    number = int(re.search(r"\d+", filename).group())
    return f"{topic}-{number:02d}"


def split_sentences(text: str) -> list[str]:
    """Split on danda, ?, ! and newlines; drop pieces that contain no word."""
    return [s.strip() for s in SENTENCE_END.split(text) if WORD.search(s)]


def build() -> list[dict]:
    rows = []
    for raw_path in sorted(RAW_DIR.glob("*/*.docx")):
        topic = raw_path.parent.name
        key = f"{topic}/{raw_path.name}"
        paragraphs = [unicodedata.normalize("NFC", p) for p in read_docx_paragraphs(raw_path)]
        notes = []

        if key in ANNOTATED:
            article, annotations = paragraphs[:1], paragraphs[1:]
            # Fail loudly if the files ever stop matching what we expect.
            assert not POS_TAG.search(article[0]), key
            assert any(POS_TAG.search(p) for p in annotations), key
            paragraphs = article
            notes.append(
                f"Only paragraph 1 extracted; the {len(annotations)} paragraph(s) after it are "
                "prototype annotations (lemma/POS/synonym lists)."
            )

        if key in DUPLICATES:
            doc_id = DUPLICATES[key] + "-copy"
            duplicate_of = DUPLICATES[key]
            path = None
            notes.append(f"Exact copy of {duplicate_of}; no .txt written.")
        else:
            doc_id = make_doc_id(topic, raw_path.name)
            duplicate_of = None
            path = f"data/corpus/{topic}/{doc_id}.txt"

        text = "\n".join(paragraphs)
        sentences = split_sentences(text)
        if len(sentences) == 1 and len(WORD.findall(text)) >= 40:
            # We report this instead of guessing boundaries (DECISIONS.md).
            notes.append("Source has no sentence-ending punctuation; whole text is one sentence.")
        if doc_id in NOTES:
            notes.append(NOTES[doc_id])
        rows.append(
            {
                "doc_id": doc_id,
                "topic": topic,
                "path": path,
                "raw_path": raw_path.relative_to(ROOT).as_posix(),
                "word_count": len(WORD.findall(text)),
                "sentence_count": len(sentences),
                "duplicate_of": duplicate_of,
                "notes": " ".join(notes),
                "_text": text,  # not written to the manifest
            }
        )

    # The duplicate really must be identical to what it claims to copy.
    texts = {row["doc_id"]: row["_text"] for row in rows}
    for row in rows:
        if row["duplicate_of"]:
            assert texts[row["doc_id"]] == texts[row["duplicate_of"]], row["doc_id"]
    ids = [row["doc_id"] for row in rows]
    assert len(ids) == len(set(ids)), "doc_id collision"
    return sorted(rows, key=lambda row: row["doc_id"])


def main() -> None:
    rows = build()
    for row in rows:
        if row["path"]:
            out = ROOT / row["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(row["_text"] + "\n", encoding="utf-8")

    with (CORPUS_DIR / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            record = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {sum(1 for r in rows if r['path'])} texts and {len(rows)} manifest rows.")


if __name__ == "__main__":
    main()
