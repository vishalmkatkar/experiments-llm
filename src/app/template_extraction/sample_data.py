"""Generate a messy multi-invoice PDF plus a ground-truth JSON file.

One file holds four invoices from three vendors, with nothing marking where one
ends and the next begins:

- Acme Logistics INV-20931: 2 pages
- Brightline Office Supply BL-55120: 2 pages, different labels and page counter
- Acme Logistics INV-20958: the long one; a line-item table across several pages,
  a wrapped row that splits across a page break, and two genuinely identical rows
- Cobalt Freight CF-3381: 1 page, a vendor with no template in the registry

The data is synthetic and cleaner than real scans. It exists to exercise the
pipeline's paths, not to benchmark accuracy.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = A4
LEFT = 40
ROW_H = 14


@dataclass
class Row:
    description: str
    quantity: float
    unit_price: float
    amount: float
    date: str | None = None
    reference: str | None = None


@dataclass
class Invoice:
    vendor: str
    invoice_number: str
    invoice_date: str
    po_number: str | None
    currency: str
    vendor_tax_id: str
    rows: list[Row]
    tax_rate: float
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    pages: list[int] = field(default_factory=list)

    def compute(self) -> None:
        self.subtotal = round(sum(r.amount for r in self.rows), 2)
        self.tax = round(self.subtotal * self.tax_rate, 2)
        self.total = round(self.subtotal + self.tax, 2)


def money(x: float) -> str:
    return f"{x:,.2f}"


def wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


# --------------------------------------------------------------------------- data


def acme_rows(rng: random.Random, n: int, start_ref: int, month: str) -> list[Row]:
    lanes = ["Hamburg-Rotterdam", "Hamburg-Antwerp", "Bremen-Gdansk", "Hamburg-Felixstowe", "Kiel-Gothenburg"]
    rows = []
    for i in range(n):
        day = 1 + i % 28
        kind = rng.random()
        if kind < 0.6:
            qty = rng.choice([1, 1, 2, 3])
            rate = rng.choice([380.0, 420.0, 455.0, 510.0, 640.0])
            desc = f"Freight charge {rng.choice(lanes)}"
        elif kind < 0.8:
            qty, rate, desc = 1, 35.0, "Fuel surcharge"
        else:
            qty = rng.choice([1, 2])
            rate = rng.choice([48.5, 62.0, 75.0])
            desc = "Customs documentation fee"
        rows.append(
            Row(desc, qty, rate, round(qty * rate, 2), date=f"2026-{month}-{day:02d}", reference=f"SHP-{start_ref + i}")
        )
    return rows


def brightline_rows(rng: random.Random, n: int) -> list[Row]:
    items = [
        ("A4 copier paper, 80gsm, box of 5 reams", 24.99),
        ("Ballpoint pens, blue, pack of 50", 11.49),
        ("Ergonomic mesh office chair", 189.00),
        ("Whiteboard markers, assorted, pack of 12", 8.75),
        ("Laptop stand, aluminium", 34.50),
        ("Recycled notebooks A5, pack of 10", 15.20),
        ("Desk lamp, LED, dimmable", 42.00),
    ]
    rows = []
    for i in range(n):
        desc, price = items[rng.randrange(len(items))]
        qty = rng.choice([1, 2, 4, 6, 10])
        rows.append(Row(desc, qty, price, round(qty * price, 2), reference=f"BL-{4100 + i}"))
    return rows


# --------------------------------------------------------------------------- rendering


class Doc:
    def __init__(self, path: Path):
        self.c = Canvas(str(path), pagesize=A4)
        self.page_no = 0

    def new_page(self) -> None:
        if self.page_no:
            self.c.showPage()
        self.page_no += 1

    def text(self, x: float, y: float, s: str, size: int = 9, bold: bool = False, right: bool = False) -> None:
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        (self.c.drawRightString if right else self.c.drawString)(x, y, s)

    def save(self) -> None:
        self.c.save()


def render_acme(doc: Doc, inv: Invoice, first_cap: int, cont_cap: int, split_row_at_break: bool) -> None:
    cols = [LEFT, 105, 170, 400, 455, 555]  # Date, Ref, Description, Qty, Rate, Amount(right)
    headers = ["Date", "Shipment Ref", "Description", "Qty", "Rate", "Amount"]

    # Lay rows out as lines; long descriptions wrap onto a second line.
    lines: list[tuple[int, int, str]] = []  # (row index, line within row, description text)
    for i, r in enumerate(inv.rows):
        for j, part in enumerate(wrap(r.description, 40)):
            lines.append((i, j, part))
    if split_row_at_break:
        # Make the row that starts on the last slot of page 1 wrap, so its second
        # line lands on page 2, below the repeated header row.
        idx = next(i for k, (i, j, _) in enumerate(lines) if k == first_cap - 1 and j == 0)
        r = inv.rows[idx]
        r.description = f"{r.description} incl. port handling and terminal storage"
        lines = [(i, j, p) for i, rr in enumerate(inv.rows) for j, p in enumerate(wrap(rr.description, 40))]

    pages: list[list[tuple[int, int, str]]] = [lines[:first_cap]]
    rest = lines[first_cap:]
    while rest:
        pages.append(rest[:cont_cap])
        rest = rest[cont_cap:]
    if len(pages[-1]) > cont_cap - 5:  # totals block needs room
        pages.append([])
    total_pages = len(pages)

    running = 0.0
    for p, page_lines in enumerate(pages, start=1):
        doc.new_page()
        inv.pages.append(doc.page_no)
        y = PAGE_H - 50
        if p == 1:
            doc.text(LEFT, y, "ACME LOGISTICS GmbH", 14, bold=True)
            doc.text(LEFT, y - 16, "Hafenstrasse 12, 20457 Hamburg, Germany")
            doc.text(LEFT, y - 28, f"VAT ID: {inv.vendor_tax_id}")
            doc.text(555, y, "INVOICE", 14, bold=True, right=True)
            doc.text(400, y - 16, f"Invoice No: {inv.invoice_number}")
            doc.text(400, y - 28, f"Invoice Date: {inv.invoice_date}")
            doc.text(400, y - 40, f"PO Number: {inv.po_number}")
            doc.text(400, y - 52, f"Currency: {inv.currency}")
            doc.text(LEFT, y - 60, "Bill to: Quillfield Labs, 1 Market Street, San Francisco, CA")
            y -= 90
        else:
            doc.text(LEFT, y, f"Acme Logistics GmbH - Invoice {inv.invoice_number} (continued)", 10, bold=True)
            y -= 24
        for x, h in zip(cols, headers):
            doc.text(x, y, h, bold=True, right=(h == "Amount"))
        y -= ROW_H
        if p > 1:
            doc.text(170, y, "Brought forward")
            doc.text(555, y, money(running), right=True)
            y -= ROW_H
        for i, j, part in page_lines:
            r = inv.rows[i]
            if j == 0:
                doc.text(cols[0], y, r.date or "")
                doc.text(cols[1], y, r.reference or "")
                doc.text(cols[2], y, part)
                doc.text(cols[3], y, f"{r.quantity:g}")
                doc.text(cols[4], y, money(r.unit_price))
                doc.text(cols[5], y, money(r.amount), right=True)
                running += r.amount
            else:
                doc.text(cols[2], y, part)
            y -= ROW_H
        if p < total_pages:
            doc.text(170, y - 6, "Carried forward", bold=True)
            doc.text(555, y - 6, money(running), right=True, bold=True)
        else:
            y -= 10
            for label, value in (("Subtotal", inv.subtotal), ("VAT 19%", inv.tax), ("Total due", inv.total)):
                doc.text(400, y, label, bold=(label == "Total due"))
                doc.text(555, y, money(value), right=True, bold=(label == "Total due"))
                y -= ROW_H
            doc.text(LEFT, y - 10, "Payment terms: 30 days net. IBAN DE44 5001 0517 5407 3249 31")
        doc.text(555, 30, f"Page {p} of {total_pages}", 8, right=True)


def render_brightline(doc: Doc, inv: Invoice, first_cap: int, cont_cap: int) -> None:
    cols = [LEFT, 100, 380, 455, 555]  # Item, Description, Units, Unit Price, Line Total(right)
    headers = ["Item", "Description", "Units", "Unit Price", "Line Total"]
    chunks = [inv.rows[:first_cap]]
    rest = inv.rows[first_cap:]
    while rest:
        chunks.append(rest[:cont_cap])
        rest = rest[cont_cap:]
    total_pages = len(chunks)
    for p, rows in enumerate(chunks, start=1):
        doc.new_page()
        inv.pages.append(doc.page_no)
        y = PAGE_H - 50
        if p == 1:
            doc.text(LEFT, y, "Brightline Office Supply Ltd", 15, bold=True)
            doc.text(LEFT, y - 15, f"Company No. 08812345 - VAT {inv.vendor_tax_id}")
            doc.text(LEFT, y - 40, f"Bill Ref: {inv.invoice_number}")
            doc.text(LEFT, y - 52, f"Date of issue: {inv.invoice_date}")
            doc.text(LEFT, y - 64, f"Your order: {inv.po_number}")
            doc.text(LEFT, y - 76, f"All amounts in {inv.currency}")
            y -= 104
        else:
            doc.text(LEFT, y, f"Brightline Office Supply Ltd - Bill Ref {inv.invoice_number} - continued", 10, bold=True)
            y -= 24
        for x, h in zip(cols, headers):
            doc.text(x, y, h, bold=True, right=(h == "Line Total"))
        y -= ROW_H
        for r in rows:
            doc.text(cols[0], y, r.reference or "")
            doc.text(cols[1], y, r.description)
            doc.text(cols[2], y, f"{r.quantity:g}")
            doc.text(cols[3], y, money(r.unit_price))
            doc.text(cols[4], y, money(r.amount), right=True)
            y -= ROW_H
        if p < total_pages:
            doc.text(LEFT, y - 6, "Continued on next sheet")
        else:
            y -= 10
            for label, value in (("Net total", inv.subtotal), ("VAT (20%)", inv.tax), ("Amount payable", inv.total)):
                doc.text(380, y, label, bold=(label == "Amount payable"))
                doc.text(555, y, money(value), right=True, bold=(label == "Amount payable"))
                y -= ROW_H
        doc.text(LEFT, 30, f"Sheet {p}/{total_pages}", 8)


def render_cobalt(doc: Doc, inv: Invoice) -> None:
    doc.new_page()
    inv.pages.append(doc.page_no)
    y = PAGE_H - 50
    doc.text(LEFT, y, "COBALT FREIGHT INC.", 13, bold=True)
    doc.text(LEFT, y - 14, "2200 Harbor Blvd, Oakland, CA 94607")
    doc.text(LEFT, y - 26, f"EIN {inv.vendor_tax_id}")
    doc.text(360, y, f"Invoice # {inv.invoice_number}", 11, bold=True)
    doc.text(360, y - 14, f"Date: {inv.invoice_date}")
    doc.text(360, y - 26, f"Customer PO: {inv.po_number}")
    y -= 64
    cols = [LEFT, 330, 420, 555]
    for x, h in zip(cols, ["Description", "Quantity", "Price", "Total"]):
        doc.text(x, y, h, bold=True, right=(h == "Total"))
    y -= ROW_H
    for r in inv.rows:
        doc.text(cols[0], y, r.description)
        doc.text(cols[1], y, f"{r.quantity:g}")
        doc.text(cols[2], y, f"${money(r.unit_price)}")
        doc.text(cols[3], y, f"${money(r.amount)}", right=True)
        y -= ROW_H
    y -= 10
    for label, value in (("Subtotal", inv.subtotal), ("Sales tax (8.5%)", inv.tax), ("Total", inv.total)):
        doc.text(420, y, label, bold=(label == "Total"))
        doc.text(555, y, f"${money(value)}", right=True, bold=(label == "Total"))
        y -= ROW_H
    doc.text(LEFT, 30, "Page 1 of 1", 8)


# --------------------------------------------------------------------------- main


def generate(out_dir: Path, seed: int = 7) -> tuple[Path, Path]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    a1 = Invoice("Acme Logistics GmbH", "INV-20931", "2026-10-03", "PO-77812", "EUR", "DE811234567", acme_rows(rng, 24, 1001, "09"), 0.19)
    b1 = Invoice("Brightline Office Supply Ltd", "BL-55120", "14 Oct 2026", "PO-77840", "GBP", "GB293447106", brightline_rows(rng, 22), 0.20)
    a2_rows = acme_rows(rng, 72, 1100, "10")
    # Two genuinely identical rows: content-only de-duplication would wrongly merge them.
    a2_rows[40] = Row("Fuel surcharge", 1, 35.0, 35.0, date="2026-10-12", reference="SHP-1140")
    a2_rows[41] = Row("Fuel surcharge", 1, 35.0, 35.0, date="2026-10-12", reference="SHP-1140")
    a2 = Invoice("Acme Logistics GmbH", "INV-20958", "2026-10-31", "PO-77901", "EUR", "DE811234567", a2_rows, 0.19)
    c1 = Invoice(
        "Cobalt Freight Inc.", "CF-3381", "10/21/2026", "PO-77866", "USD", "94-3318820",
        [
            Row("Drayage Oakland port to warehouse", 3, 285.0, 855.0),
            Row("Chassis rental (per day)", 6, 42.0, 252.0),
            Row("Pre-pull and yard storage", 2, 95.0, 190.0),
            Row("Fuel surcharge", 3, 31.5, 94.5),
            Row("Container inspection", 1, 160.0, 160.0),
            Row("Waiting time (per hour)", 2, 85.0, 170.0),
        ],
        0.085,
    )

    for inv in (a1, b1, a2, c1):
        inv.compute()  # totals don't depend on layout, so compute them before rendering

    pdf_path = out_dir / "vendor_bundle.pdf"
    doc = Doc(pdf_path)
    render_acme(doc, a1, first_cap=16, cont_cap=34, split_row_at_break=False)
    render_brightline(doc, b1, first_cap=14, cont_cap=34)
    render_acme(doc, a2, first_cap=22, cont_cap=30, split_row_at_break=True)
    render_cobalt(doc, c1)
    doc.save()

    truth_path = out_dir / "vendor_bundle.truth.json"
    truth_path.write_text(json.dumps([asdict(i) for i in (a1, b1, a2, c1)], indent=2), encoding="utf-8")
    return pdf_path, truth_path


if __name__ == "__main__":
    pdf, truth = generate(Path(__file__).parent / "samples")
    print(f"Wrote {pdf}\nWrote {truth}")
