import hashlib

from hindi_plag.corpus import DOMAINS, KNOWN_DUPLICATES, default_corpus_dir, load_corpus

CORPUS_DIR = default_corpus_dir()


def test_corpus_files_are_unmodified() -> None:
    """Every .docx matches data/corpus/SHA256SUMS, and nothing is missing or extra."""
    expected: dict[str, str] = {}
    for line in (CORPUS_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest

    on_disk = {p.relative_to(CORPUS_DIR).as_posix() for p in CORPUS_DIR.rglob("*.docx")}
    assert on_disk == set(expected)
    for name, digest in expected.items():
        assert hashlib.sha256((CORPUS_DIR / name).read_bytes()).hexdigest() == digest, name


def test_domain_folders_match_constant() -> None:
    folders = {p.name for p in CORPUS_DIR.iterdir() if p.is_dir()}
    assert folders == set(DOMAINS)


def test_document_counts() -> None:
    docs = load_corpus(include_duplicates=True)
    counts = {d: sum(doc.domain == d for doc in docs) for d in DOMAINS}
    assert counts == {"mahabharat": 21, "nisarga": 25, "paryatan": 10, "ramayan": 9}
    assert len(load_corpus()) == 65 - len(KNOWN_DUPLICATES)


def test_every_document_has_hindi_text() -> None:
    for doc in load_corpus():
        devanagari = sum("ऀ" <= ch <= "ॿ" for ch in doc.text)
        assert devanagari > 200, doc.doc_id


def test_known_duplicates_are_exact_and_complete() -> None:
    by_id = {doc.doc_id: doc.text for doc in load_corpus(include_duplicates=True)}
    for duplicate, original in KNOWN_DUPLICATES.items():
        assert by_id[duplicate] == by_id[original]

    # No *undeclared* exact duplicates once the known ones are skipped.
    texts = [doc.text for doc in load_corpus()]
    assert len(texts) == len(set(texts))
