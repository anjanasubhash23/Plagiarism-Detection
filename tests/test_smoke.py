import json
from pathlib import Path

import srot

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_and_corpus_manifest_matches_files() -> None:
    assert srot.__version__

    lines = (ROOT / "data/corpus/manifest.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    sources = [r for r in rows if r["duplicate_of"] is None]
    assert len(rows) == 65 and len(sources) == 64

    for r in rows:
        assert (ROOT / r["raw_path"]).exists(), r["raw_path"]
    for r in sources:
        assert (ROOT / r["path"]).read_text(encoding="utf-8").strip(), r["doc_id"]
    assert {p.relative_to(ROOT).as_posix() for p in ROOT.glob("data/corpus/*/*.txt")} == {
        r["path"] for r in sources
    }
