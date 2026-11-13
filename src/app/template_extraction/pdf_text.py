"""Read the PDF text layer as lines, with column boundaries marked by ' | '.

Digital PDFs carry exact text with coordinates, which beats OCR on a screenshot:
no misread digits, and anchors like "Subtotal" can be matched as text.
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from .config import COLUMN_GAP


@dataclass(frozen=True)
class Line:
    text: str
    top: float


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, within the file
    lines: tuple[Line, ...]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def _join(words: list[dict]) -> str:
    parts = [words[0]["text"]]
    for prev, cur in zip(words, words[1:]):
        parts.append(" | " if cur["x0"] - prev["x1"] > COLUMN_GAP else " ")
        parts.append(cur["text"])
    return "".join(parts)


def load_pages(path: Path) -> list[Page]:
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            words = sorted(page.extract_words(), key=lambda w: (round(w["top"]), w["x0"]))
            groups: list[list[dict]] = []
            for w in words:
                if groups and abs(groups[-1][0]["top"] - w["top"]) < 3:
                    groups[-1].append(w)
                else:
                    groups.append([w])
            lines = tuple(Line(_join(sorted(g, key=lambda w: w["x0"])), g[0]["top"]) for g in groups)
            pages.append(Page(number, lines))
    return pages
