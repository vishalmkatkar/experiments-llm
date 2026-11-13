"""Steps 1 and 2: classify pages, split the file into documents, find sections.

No LLM calls here. With a template, page roles, document ids and section
anchors are all rules, which makes this part fast, cheap and debuggable.
"""

import json
import re
from dataclasses import dataclass, field

from .config import CHUNK_OVERLAP, CHUNK_ROWS, REGISTRY_DIR
from .pdf_text import Page

MONEY_AT_END = re.compile(r"[$£€]?\d{1,3}(?:,\d{3})*\.\d{2}\s*$")
GENERIC_DOC_ID = re.compile(r"(?:Invoice|Bill)\s*(?:No\.?|Number|#|Ref)\s*[:#]?\s*([A-Z]{1,4}-?\d{3,})", re.IGNORECASE)
GENERIC_COUNTER = re.compile(r"(?:Page|Sheet)\s*(\d+)\s*(?:of|/)\s*(\d+)", re.IGNORECASE)
TOTALS_WORDS = re.compile(r"\b(sub\s*total|net total|tax|vat|total|amount payable|amount due)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- registry


def load_registry() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(REGISTRY_DIR.glob("*.json"))]


def match_template(page: Page, templates: list[dict]) -> dict | None:
    text = page.text.lower()
    for t in templates:
        fp = t["fingerprint"]
        if any(tid.lower() in text for tid in fp["tax_ids"]) or any(k.lower() in text for k in fp["keywords"]):
            return t
    return None


# --------------------------------------------------------------------------- step 1: classify + split


@dataclass
class PageInfo:
    number: int
    template_id: str | None
    role: str  # first | continuation
    doc_id: str | None
    counter: tuple[int, int] | None


@dataclass
class DocGroup:
    pages: list[int]
    template_id: str | None
    doc_id: str | None
    counters: list[tuple[int, int] | None] = field(default_factory=list)


def classify_page(page: Page, templates: list[dict]) -> PageInfo:
    t = match_template(page, templates)
    text = page.text
    if t:
        m = re.search(t["document_id"]["pattern"], text)
        c = re.search(t["page_counter"], text)
        is_first = any(s in text for s in t["page_roles"]["first"]["contains"])
        return PageInfo(
            page.number, t["id"], "first" if is_first else "continuation",
            m.group(0) if m else None, (int(c.group(1)), int(c.group(2))) if c else None,
        )
    m = GENERIC_DOC_ID.search(text)
    c = GENERIC_COUNTER.search(text)
    counter = (int(c.group(1)), int(c.group(2))) if c else None
    is_first = bool(m) or (counter is not None and counter[0] == 1)
    return PageInfo(page.number, None, "first" if is_first else "continuation", m.group(1) if m else None, counter)


def split_documents(infos: list[PageInfo]) -> list[DocGroup]:
    docs: list[DocGroup] = []
    for info in infos:
        cur = docs[-1] if docs else None
        starts_new = (
            cur is None
            or info.role == "first"
            or info.template_id != cur.template_id
            or (info.doc_id is not None and cur.doc_id is not None and info.doc_id != cur.doc_id)
        )
        if starts_new:
            docs.append(DocGroup([info.number], info.template_id, info.doc_id, [info.counter]))
        else:
            cur.pages.append(info.number)
            cur.counters.append(info.counter)
            cur.doc_id = cur.doc_id or info.doc_id
    return docs


# --------------------------------------------------------------------------- step 2: sections


@dataclass
class TableRow:
    id: str
    page: int
    cells: list[str]
    rejoined: bool = False  # a wrapped line (possibly from the next page) was joined back

    @property
    def text(self) -> str:
        return " | ".join(self.cells)


@dataclass
class Sections:
    header_text: str
    rows: list[TableRow]
    totals_text: str
    notes: list[str]


def locate_sections(pages: list[Page], template: dict) -> Sections:
    li = template["sections"]["line_items"]
    desc_idx = li["columns"].index("description")
    counter = re.compile(template["page_counter"])
    header_lines: list[str] = []
    rows: list[TableRow] = []
    rejoined = 0

    for idx, page in enumerate(pages):
        in_table = False
        for line in page.lines:
            text = line.text
            if not in_table:
                if all(h in text for h in li["header_row_contains"]):
                    in_table = True  # the header row repeats on every continuation page
                elif idx == 0 and not rows:
                    header_lines.append(text)
                continue
            if any(a in text for a in li["end_anchors"]):
                in_table = False
                continue
            if any(s in text for s in li["skip_lines"]) or counter.search(text):
                continue
            if MONEY_AT_END.search(text):
                rows.append(TableRow(f"r{len(rows) + 1:03d}", page.number, text.split(" | ")))
            elif rows:
                # A line with no amount continues the previous row's description.
                # This also rejoins a row that was split across a page break.
                prev = rows[-1]
                target = desc_idx if len(prev.cells) == len(li["columns"]) else -1
                prev.cells[target] = f"{prev.cells[target]} {text.strip()}"
                prev.rejoined = True
                rejoined += 1

    labels = list(template["sections"]["totals"]["fields"].values())
    last = pages[-1]
    totals_lines = [line.text for line in last.lines if any(line.text.startswith(lb) for lb in labels)]

    notes = [f"{len(rows)} rows detected on {len(pages)} page(s)"]
    if rejoined:
        cross = [r for r in rows if r.rejoined]
        notes.append(f"rejoined {rejoined} wrapped line(s) into {', '.join(r.id for r in cross)}")
    return Sections("\n".join(header_lines), rows, "\n".join(totals_lines), notes)


def read_totals(totals_text: str, template: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for field_name, label in template["sections"]["totals"]["fields"].items():
        m = re.search(re.escape(label) + r"\s*\|?\s*[$£€]?([\d,]+\.\d{2})", totals_text)
        out[field_name] = float(m.group(1).replace(",", "")) if m else None
    return out


def parse_row_by_rules(row: TableRow, columns: list[str]) -> dict | None:
    """Fast path for clean digital PDFs: if a row's cells line up exactly with the
    template's columns, parse it directly. Returns None when the row is ragged,
    so it goes to the LLM instead."""
    if len(row.cells) != len(columns):
        return None
    out: dict = {"row_id": row.id}
    for col, cell in zip(columns, row.cells):
        if col in ("quantity", "unit_price", "amount"):
            try:
                out[col] = float(cell.strip().lstrip("$£€").replace(",", ""))
            except ValueError:
                return None
        else:
            out[col] = cell.strip()
    return out


def make_chunks(rows: list[TableRow], size: int = CHUNK_ROWS, overlap: int = CHUNK_OVERLAP) -> list[list[TableRow]]:
    if len(rows) <= size:
        return [rows]
    chunks, start = [], 0
    while True:
        chunks.append(rows[start : start + size])
        if start + size >= len(rows):
            return chunks
        start += size - overlap


def count_generic_rows(pages: list[Page]) -> int:
    """Rough row count for documents with no template: lines ending in an amount,
    excluding totals lines. Used only as a sanity check on fallback extraction."""
    return sum(
        1
        for p in pages
        for line in p.lines
        if MONEY_AT_END.search(line.text) and not TOTALS_WORDS.search(line.text.split(" | ")[0])
    )
