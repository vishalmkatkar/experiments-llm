"""Two LangGraph graphs.

File graph:      load_pdf -> classify_pages -> split_documents -> process_document (one Send per document)
Document graph:  template known?  locate_sections -> extract_header -> extract_chunk (one Send per chunk)
                                  -> stitch -> validate
                 unknown vendor:  generic_extract -> validate -> propose_template (if the checks pass)
"""

import json
import operator
import re
import time
from collections import Counter
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import extract
from .config import CHUNK_OVERLAP, OUT_DIR
from .layout import (
    DocGroup, PageInfo, Sections, TableRow, classify_page, count_generic_rows, load_registry,
    locate_sections, make_chunks, parse_row_by_rules, read_totals, split_documents,
)
from .pdf_text import Page, load_pages
from .validate import run_checks

TEMPLATES = {t["id"]: t for t in load_registry()}


# =========================================================================== document graph


class DocState(TypedDict, total=False):
    doc: DocGroup
    pages: list[Page]
    template: dict | None
    drop_row: bool  # fault injection: simulate the model skipping one row
    mode: str  # "llm" (every row through the LLM) | "fast" (rules first, LLM for ragged rows)
    sections: Sections
    parsed_items: list[dict]
    chunks: list[list[TableRow]]
    header: dict
    totals: dict
    chunk_results: Annotated[list[dict], operator.add]
    items: list[dict]
    expected_rows: int | None
    overlap_conflicts: list[str]
    checks: list[dict]
    status: str  # approved | needs_review
    candidate: str | None
    llm_calls: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


def route_by_template(s: DocState) -> str:
    return "locate_sections" if s.get("template") else "generic_extract"


def locate_sections_node(s: DocState) -> dict:
    t = s["template"]
    sections = locate_sections(s["pages"], t)
    columns = t["sections"]["line_items"]["columns"]
    parsed: list[dict] = []
    pending = sections.rows
    trace = [f"locate_sections: {'; '.join(sections.notes)}"]
    if s.get("mode") == "fast":
        pending = []
        for row in sections.rows:
            p = parse_row_by_rules(row, columns)
            parsed.append(p) if p else pending.append(row)
        trace.append(f"fast mode: rules parsed {len(parsed)} row(s) from the text layer; {len(pending)} left for the LLM")
    chunks = make_chunks(pending) if pending else []
    trace += [f"chunking: {len(chunks)} chunk(s) for the LLM", "read_totals: from template labels (no LLM call)"]
    return {
        "sections": sections,
        "parsed_items": parsed,
        "chunks": chunks,
        "totals": read_totals(sections.totals_text, t),
        "expected_rows": len(sections.rows),
        "trace": trace,
    }


def extract_header_node(s: DocState) -> dict:
    header = extract.extract_header(s["sections"].header_text, s["template"])
    return {"header": header.model_dump(), "llm_calls": 1, "trace": [f"extract_header: {header.invoice_number}, {header.currency}"]}


def fan_out_chunks(s: DocState) -> list[Send] | str:
    chunks = s["chunks"]
    if not chunks:
        return "stitch"  # fast mode parsed every row; nothing for the LLM to do
    header = extract.Header(**s["header"])
    columns = s["template"]["sections"]["line_items"]["columns"]
    return [
        Send("extract_chunk", {
            "rows": chunk, "index": i, "count": len(chunks), "header": header, "columns": columns,
            "overlap": min(len(chunk), CHUNK_OVERLAP) if i else 0,
        })
        for i, chunk in enumerate(chunks)
    ]


def extract_chunk_node(task: dict) -> dict:
    rows: list[TableRow] = task["rows"]
    out = extract.extract_chunk(rows, task["index"], task["count"], task["header"], task["columns"], task["overlap"])
    items = [r.model_dump() for r in out.rows]
    return {
        "chunk_results": [{"index": task["index"], "rows": items}],
        "llm_calls": 1,
        "trace": [f"extract_chunk {task['index'] + 1}/{task['count']}: {len(items)} rows ({rows[0].id}-{rows[-1].id})"],
    }


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def stitch_node(s: DocState) -> dict:
    by_id: dict[str, dict] = {r["row_id"]: r for r in s.get("parsed_items", [])}
    conflicts: list[str] = []
    duplicates = 0
    for result in sorted(s.get("chunk_results", []), key=lambda r: r["index"]):
        for row in result["rows"]:
            rid = row["row_id"]
            if rid in by_id:
                # Same position extracted twice (chunk overlap): keep one, but both copies must agree.
                duplicates += 1
                a = by_id[rid]
                if any(a[k] != row[k] for k in ("quantity", "unit_price", "amount")) or _norm(a["description"]) != _norm(row["description"]):
                    conflicts.append(rid)
            else:
                by_id[rid] = row

    order = [r.id for r in s["sections"].rows]
    items = [by_id[i] for i in order if i in by_id]
    unknown = sorted(set(by_id) - set(order))
    content = Counter(
        (_norm(i["description"]), i.get("date"), i.get("reference"), i["quantity"], i["unit_price"], i["amount"]) for i in items
    )
    identical = sum(c - 1 for c in content.values() if c > 1)
    sources = []
    if s.get("parsed_items"):
        sources.append(f"{len(s['parsed_items'])} rule-parsed")
    if s.get("chunk_results"):
        sources.append(f"{len(s['chunk_results'])} LLM chunk(s)")
    note = f"stitch: {len(items)} rows from {' + '.join(sources) or 'nothing'}; {duplicates} overlap row(s) de-duplicated by row id"
    if identical:
        note += f"; kept {identical} row(s) identical in content to another (content-only matching would have merged them)"
    if unknown:
        note += f"; ignored unknown row ids {unknown}"
    trace = [note]
    if s.get("drop_row") and len(items) > 30:
        dropped = items.pop(30)
        trace.append(f"FAULT INJECTED: removed {dropped['row_id']} to simulate the model skipping a row")
    return {"items": items, "overlap_conflicts": conflicts, "trace": trace}


def generic_extract_node(s: DocState) -> dict:
    text = "\n".join(p.text for p in s["pages"])
    full = extract.extract_full(text)
    items = [{"row_id": f"g{i + 1:03d}", **r.model_dump()} for i, r in enumerate(full.rows)]
    return {
        "header": full.header.model_dump(),
        "items": items,
        "totals": {"subtotal": full.subtotal, "tax": full.tax, "total": full.total},
        "expected_rows": count_generic_rows(s["pages"]),
        "overlap_conflicts": [],
        "llm_calls": 1,
        "trace": [f"generic_extract: no template for this vendor; whole-document extraction, {len(items)} rows"],
    }


def validate_node(s: DocState) -> dict:
    required = (s.get("template") or {}).get("validation", {}).get("required", ["invoice_number", "invoice_date", "total"])
    checks = run_checks(
        s.get("header", {}), s.get("items", []), s.get("totals", {}), required,
        s.get("expected_rows"), s["doc"].counters, s.get("overlap_conflicts", []),
    )
    failed = [c["name"] for c in checks if not c["ok"]]
    status = "needs_review" if failed else "approved"
    return {"checks": checks, "status": status, "trace": [f"validate: {status}" + (f" (failed: {', '.join(failed)})" if failed else "")]}


def after_validate(s: DocState) -> str:
    return "propose_template" if not s.get("template") and s.get("status") == "approved" else END


def propose_template_node(s: DocState) -> dict:
    """Templates are a cache: a validated result from an unknown vendor becomes a
    draft template. A person reviews it before it joins the registry."""
    header = s["header"]
    text = "\n".join(p.text for p in s["pages"])
    lines = text.splitlines()
    header_row = next((ln for ln in lines if "description" in ln.lower()), "")
    totals_labels = [ln.split(" | ")[0] for ln in lines if re.match(r"(sub\s*total|.*tax|total)\b", ln, re.IGNORECASE)]
    slug = re.sub(r"[^a-z0-9]+", "_", header["vendor_name"].lower()).strip("_")
    candidate = {
        "id": slug,
        "status": "candidate: needs human review before joining the registry",
        "vendor": header["vendor_name"],
        "document_type": "invoice",
        "currency": header["currency"],
        "fingerprint": {"tax_ids": [header["vendor_tax_id"]] if header.get("vendor_tax_id") else [], "keywords": [header["vendor_name"]]},
        "document_id": {"pattern": re.sub(r"\d", r"\\d", re.escape(header["invoice_number"]))},
        "page_counter": r"Page (\d+) of (\d+)",
        "sections": {
            "line_items": {
                "header_row_contains": [c.strip() for c in header_row.split(" | ")],
                "end_anchors": totals_labels[:1],
            },
            "totals": {"fields": dict(zip(("subtotal", "tax", "total"), totals_labels))},
        },
    }
    path = OUT_DIR / "candidates" / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    return {"candidate": str(path), "trace": [f"propose_template: validated, so drafted a template candidate for review: {path.name}"]}


def build_document_graph():
    g = StateGraph(DocState)
    g.add_node("locate_sections", locate_sections_node)
    g.add_node("extract_header", extract_header_node)
    g.add_node("extract_chunk", extract_chunk_node)
    g.add_node("stitch", stitch_node)
    g.add_node("generic_extract", generic_extract_node)
    g.add_node("validate", validate_node)
    g.add_node("propose_template", propose_template_node)

    g.add_conditional_edges(START, route_by_template, ["locate_sections", "generic_extract"])
    g.add_edge("locate_sections", "extract_header")
    g.add_conditional_edges("extract_header", fan_out_chunks, ["extract_chunk", "stitch"])
    g.add_edge("extract_chunk", "stitch")  # runs once, after every chunk branch finishes
    g.add_edge("stitch", "validate")
    g.add_edge("generic_extract", "validate")
    g.add_conditional_edges("validate", after_validate, ["propose_template", END])
    g.add_edge("propose_template", END)
    return g.compile()


DOCUMENT_GRAPH = build_document_graph()


# =========================================================================== file graph


class FileState(TypedDict, total=False):
    pdf_path: str
    only: list[str] | None
    drop_row_in: str | None
    mode: str
    pages: list[Page]
    infos: list[PageInfo]
    docs: list[DocGroup]
    results: Annotated[list[dict], operator.add]
    trace: Annotated[list[str], operator.add]


def load_pdf_node(s: FileState) -> dict:
    pages = load_pages(s["pdf_path"])
    return {"pages": pages, "trace": [f"load_pdf: {len(pages)} pages from the text layer"]}


def classify_pages_node(s: FileState) -> dict:
    infos = [classify_page(p, list(TEMPLATES.values())) for p in s["pages"]]
    lines = [
        f"classify p{i.number}: {i.template_id or 'unknown vendor'}, {i.role}, {i.doc_id or '?'}"
        + (f", {i.counter[0]}/{i.counter[1]}" if i.counter else "")
        for i in infos
    ]
    return {"infos": infos, "trace": lines}


def split_documents_node(s: FileState) -> dict:
    docs = split_documents(s["infos"])
    if s.get("only"):
        docs = [d for d in docs if d.doc_id in s["only"]]
    summary = ", ".join(f"{d.doc_id} (p{d.pages[0]}-{d.pages[-1]})" for d in docs)
    return {"docs": docs, "trace": [f"split: {summary}"]}


def fan_out_documents(s: FileState) -> list[Send]:
    return [
        Send("process_document", {
            "doc": d,
            "pages": [s["pages"][n - 1] for n in d.pages],
            "template": TEMPLATES.get(d.template_id) if d.template_id else None,
            "drop_row": d.doc_id == s.get("drop_row_in"),
            "mode": s.get("mode", "llm"),
        })
        for d in s["docs"]
    ]


def process_document_node(task: dict) -> dict:
    started = time.monotonic()
    out = DOCUMENT_GRAPH.invoke({**task, "llm_calls": 0, "trace": [], "chunk_results": []})
    doc: DocGroup = task["doc"]
    return {"results": [{
        "doc_id": doc.doc_id,
        "template_id": doc.template_id,
        "pages": doc.pages,
        "header": out.get("header", {}),
        "items": out.get("items", []),
        "totals": out.get("totals", {}),
        "checks": out.get("checks", []),
        "status": out.get("status"),
        "candidate": out.get("candidate"),
        "chunks": len(out.get("chunks", [])) or None,
        "llm_calls": out.get("llm_calls", 0),
        "trace": out.get("trace", []),
        "elapsed_s": round(time.monotonic() - started, 1),
    }]}


def build_file_graph():
    g = StateGraph(FileState)
    g.add_node("load_pdf", load_pdf_node)
    g.add_node("classify_pages", classify_pages_node)
    g.add_node("split_documents", split_documents_node)
    g.add_node("process_document", process_document_node)
    g.add_edge(START, "load_pdf")
    g.add_edge("load_pdf", "classify_pages")
    g.add_edge("classify_pages", "split_documents")
    g.add_conditional_edges("split_documents", fan_out_documents, ["process_document"])
    g.add_edge("process_document", END)
    return g.compile()


FILE_GRAPH = build_file_graph()
