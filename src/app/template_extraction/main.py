"""Run template-guided, section-wise extraction on a multi-invoice PDF.

    cd src
    ../.venv/bin/python -m app.template_extraction.main                       # whole sample bundle
    ../.venv/bin/python -m app.template_extraction.main --only INV-20958      # one document
    ../.venv/bin/python -m app.template_extraction.main --fast                # rules for clean rows, LLM for the rest
    ../.venv/bin/python -m app.template_extraction.main --drop-row INV-20958  # watch validation catch a skipped row
    ../.venv/bin/python -m app.template_extraction.main --generate            # rebuild the sample PDF
    ../.venv/bin/python -m app.template_extraction.main --mermaid graphs/     # export both graphs
"""

import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from .config import SAMPLES_DIR

DEFAULT_PDF = SAMPLES_DIR / "vendor_bundle.pdf"
HEADER_FIELDS = ("invoice_number", "invoice_date", "po_number", "currency", "vendor_tax_id")


def _norm(v) -> str:
    return re.sub(r"[\s\-.]", "", str(v or "")).lower()


def _row_key(r: dict) -> tuple:
    return (_norm(r["description"]), round(r["quantity"], 4), round(r["unit_price"], 2), round(r["amount"], 2))


def score(result: dict, truth: dict) -> str:
    """Compare against the generator's ground truth. Only possible on synthetic data."""
    header = result["header"]
    wrong = [f"{f}={header.get(f)!r} (expected {truth[f]!r})" for f in HEADER_FIELDS if _norm(header.get(f)) != _norm(truth[f])]
    # Align rows as sequences, so one missing row doesn't make every later row look wrong.
    matcher = SequenceMatcher(None, [_row_key(r) for r in result["items"]], [_row_key(r) for r in truth["rows"]], autojunk=False)
    rows_ok = sum(b.size for b in matcher.get_matching_blocks())
    out = f"header {len(HEADER_FIELDS) - len(wrong)}/{len(HEADER_FIELDS)} fields, rows {rows_ok}/{len(truth['rows'])} exact"
    if wrong:
        out += "\n    header differences: " + "; ".join(wrong)
    return out


def show(result: dict, truth: dict | None, verbose: bool) -> None:
    vendor = result["header"].get("vendor_name", "?")
    tmpl = result["template_id"] or "no template"
    pages = f"p{result['pages'][0]}-{result['pages'][-1]}" if len(result["pages"]) > 1 else f"p{result['pages'][0]}"
    print(f"\n━━ {result['doc_id']} · {vendor} · {tmpl} · {pages} ━━")
    if verbose:
        for line in result["trace"]:
            print(f"   · {line}")
    for c in result["checks"]:
        print(f"   {'✓' if c['ok'] else '✗'} {c['name']:<18} {c['detail']}")
    print(f"→ {result['status']} | LLM calls: {result['llm_calls']} | {result['elapsed_s']}s")
    if truth:
        print(f"  Ground truth: {score(result, truth)}")
    if result.get("candidate"):
        print(f"  Template candidate: {result['candidate']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF to process (default: the sample bundle)")
    parser.add_argument("--only", help="comma-separated document ids to process, e.g. INV-20958,CF-3381")
    parser.add_argument("--drop-row", metavar="DOC_ID", help="simulate the model skipping one row in this document")
    parser.add_argument("--fast", action="store_true", help="parse clean rows by rules from the text layer; send only ragged rows to the LLM")
    parser.add_argument("--generate", action="store_true", help="regenerate the sample PDF and ground truth")
    parser.add_argument("--mermaid", metavar="DIR", type=Path, help="write both graphs as Mermaid files and exit")
    parser.add_argument("--quiet", action="store_true", help="hide the per-step trace")
    args = parser.parse_args(argv)

    if args.mermaid:
        from .graph import DOCUMENT_GRAPH, FILE_GRAPH

        args.mermaid.mkdir(parents=True, exist_ok=True)
        (args.mermaid / "file_graph.mmd").write_text(FILE_GRAPH.get_graph().draw_mermaid(), encoding="utf-8")
        (args.mermaid / "document_graph.mmd").write_text(DOCUMENT_GRAPH.get_graph().draw_mermaid(), encoding="utf-8")
        print(f"Wrote {args.mermaid / 'file_graph.mmd'} and {args.mermaid / 'document_graph.mmd'}")
        return 0

    if args.generate or (args.pdf == DEFAULT_PDF and not DEFAULT_PDF.exists()):
        from .sample_data import generate

        pdf, truth_path = generate(SAMPLES_DIR)
        print(f"Generated {pdf} and {truth_path}", file=sys.stderr)

    from .graph import FILE_GRAPH

    truth_path = args.pdf.with_suffix(".truth.json")
    truth = {t["invoice_number"]: t for t in json.loads(truth_path.read_text())} if truth_path.exists() else {}

    started = time.monotonic()
    out = FILE_GRAPH.invoke({
        "pdf_path": str(args.pdf),
        "only": [d.strip() for d in args.only.split(",")] if args.only else None,
        "drop_row_in": args.drop_row,
        "mode": "fast" if args.fast else "llm",
        "results": [],
        "trace": [],
    })
    if not args.quiet:
        print(f"File: {args.pdf.name}")
        for line in out["trace"]:
            print(f"   · {line}")

    results = sorted(out["results"], key=lambda r: r["pages"][0])
    for r in results:
        show(r, truth.get(r["doc_id"]), not args.quiet)

    print(f"\n{'Document':<11} {'Template':<18} {'Rows':>5} {'LLM':>4}  Status")
    for r in results:
        print(f"{r['doc_id']:<11} {r['template_id'] or '(none)':<18} {len(r['items']):>5} {r['llm_calls']:>4}  {r['status']}")
    print(f"Total: {sum(r['llm_calls'] for r in results)} LLM calls, {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
