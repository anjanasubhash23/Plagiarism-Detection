"""Load the Hindi source corpus from data/corpus/<domain>/*.docx."""

from dataclasses import dataclass
from pathlib import Path

from hindi_plag.docx import read_docx_paragraphs

DOMAINS: tuple[str, ...] = ("mahabharat", "nisarga", "paryatan", "ramayan")

# Exact copies that exist in the original repo. We never delete corpus files
# (DECISIONS.md D2), so duplicates are declared here and skipped by default;
# otherwise they would leak into the benchmark as fake "plagiarism".
# Maps duplicate doc_id -> the doc_id it copies. Verified by a test.
KNOWN_DUPLICATES: dict[str, str] = {
    "mahabharat/Copy of doc_7": "mahabharat/doc_7",
}


@dataclass(frozen=True)
class Document:
    doc_id: str  # "<domain>/<file stem>", e.g. "ramayan/Doc1"
    domain: str
    path: Path
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def default_corpus_dir() -> Path:
    """data/corpus at the repository root (this file is src/hindi_plag/corpus.py)."""
    return Path(__file__).resolve().parents[2] / "data" / "corpus"


def load_corpus(
    corpus_dir: Path | None = None, include_duplicates: bool = False
) -> list[Document]:
    """Load every .docx under each domain folder, sorted by doc_id."""
    corpus_dir = corpus_dir or default_corpus_dir()
    documents: list[Document] = []
    for domain in DOMAINS:
        for path in sorted((corpus_dir / domain).glob("*.docx")):
            doc_id = f"{domain}/{path.stem}"
            if doc_id in KNOWN_DUPLICATES and not include_duplicates:
                continue
            documents.append(
                Document(
                    doc_id=doc_id,
                    domain=domain,
                    path=path,
                    paragraphs=tuple(read_docx_paragraphs(path)),
                )
            )
    return documents
