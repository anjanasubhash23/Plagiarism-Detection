"""Extract plain-text paragraphs from a .docx file.

A .docx is a zip archive; the body text lives in ``word/document.xml``.
We only need paragraph text, so the standard library (zipfile + ElementTree)
is enough and we avoid a python-docx dependency. See DECISIONS.md (D4).
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# WordprocessingML namespace, written the way ElementTree expects it.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx_paragraphs(path: Path) -> list[str]:
    """Return the non-empty paragraphs of a .docx, in document order.

    Text is returned as written (no Unicode normalisation, no stripping of
    inner whitespace) so the loader stays faithful to the source; cleaning
    is a separate, explicit step in later phases.
    """
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_W}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t":
                parts.append(node.text or "")
            elif node.tag in (f"{_W}tab", f"{_W}br"):
                # Tabs and line breaks inside a paragraph separate words.
                parts.append(" ")
        text = "".join(parts)
        if text.strip():
            paragraphs.append(text)
    return paragraphs
