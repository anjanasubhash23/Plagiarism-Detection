import zipfile
from pathlib import Path

from hindi_plag.docx import read_docx_paragraphs

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(path: Path, body_xml: str) -> Path:
    """Write a minimal .docx containing only word/document.xml."""
    xml = f'<w:document xmlns:w="{W_NS}"><w:body>{body_xml}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return path


def test_joins_runs_and_skips_empty_paragraphs(tmp_path: Path) -> None:
    body = (
        "<w:p><w:r><w:t>राम ने </w:t></w:r><w:r><w:t>वन देखा</w:t></w:r></w:p>"
        "<w:p></w:p>"
        "<w:p><w:r><w:t>   </w:t></w:r></w:p>"
        "<w:p><w:r><w:t>सीता</w:t></w:r></w:p>"
    )
    path = make_docx(tmp_path / "a.docx", body)
    assert read_docx_paragraphs(path) == ["राम ने वन देखा", "सीता"]


def test_tabs_and_breaks_become_spaces(tmp_path: Path) -> None:
    body = "<w:p><w:r><w:t>एक</w:t><w:tab/><w:t>दो</w:t><w:br/><w:t>तीन</w:t></w:r></w:p>"
    path = make_docx(tmp_path / "b.docx", body)
    assert read_docx_paragraphs(path) == ["एक दो तीन"]
