"""Step 3: section-wise extraction. One narrow question per LLM call."""

from functools import lru_cache

from pydantic import BaseModel, Field, create_model

from . import llm
from .layout import TableRow


class Header(BaseModel):
    vendor_name: str
    invoice_number: str = Field(description="The invoice's own number or reference, e.g. after 'Invoice No', 'Invoice #' or 'Bill Ref'")
    invoice_date: str = Field(description="Issue date, exactly as printed")
    # Plain strings rather than `str | None`: with a nullable type in the schema, a
    # small model tends to answer null even when the value is printed.
    po_number: str = Field(description="The customer's purchase order number, e.g. after 'PO Number', 'Customer PO' or 'Your order'; empty string if absent")
    currency: str = Field(description="3-letter ISO currency code; infer USD from $, GBP from £, EUR from €")
    vendor_tax_id: str = Field(description="The vendor's tax or employer id (VAT ID, EIN, tax ID) without its label; empty string if absent")


_COLUMN_TYPES = {
    # Plain strings, not `str | None`: a nullable type makes small models drop printed values.
    "date": (str, Field(description="The row's date exactly as printed")),
    "reference": (str, Field(description="The row's reference or item code exactly as printed, e.g. SHP-1001 or BL-4100")),
    "description": (str, ...),
    "quantity": (float, ...),
    "unit_price": (float, ...),
    "amount": (float, ...),
}


@lru_cache
def chunk_schema(columns: tuple[str, ...]) -> type[BaseModel]:
    """Output schema with only this template's columns. Every key is repeated on
    every row, so dropping unused columns cuts output tokens on long tables."""
    row = create_model("LineItem", row_id=(str, ...), **{c: _COLUMN_TYPES[c] for c in columns})
    return create_model("ChunkRows", rows=(list[row], ...))


class GenericItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class FullInvoice(BaseModel):
    header: Header
    rows: list[GenericItem]
    subtotal: float
    tax: float
    total: float


HEADER_SYSTEM = """You extract invoice header fields from the header section of one invoice.
Use the field hints to find each value. Return values exactly as printed: do not reformat dates or codes.
Use the 3-letter ISO currency code. Use null for a field that is not present."""

CHUNK_SYSTEM = """You extract line items from one chunk of an invoice table.
Each input row starts with its row id (like r012), followed by cells separated by " | ".
Return exactly one object per input row id, in the same order, as printed.
Do not skip, merge, reorder, summarise or calculate anything.
For numbers, remove thousands separators and currency symbols. Use null for cells a row doesn't have."""

GENERIC_SYSTEM = """You extract a complete invoice from its text. There is no template for this vendor.
Cells in a line are separated by " | ". Return the header fields, every line item exactly as printed
(do not skip, merge or calculate), and the subtotal, tax and total as printed.
For numbers, remove thousands separators and currency symbols. Use the 3-letter ISO currency code."""


def extract_header(header_text: str, template: dict) -> Header:
    hints = "\n".join(f"- {k}: {v}" for k, v in template["field_hints"].items())
    user = f"Vendor: {template['vendor']}\nField hints:\n{hints}\n\nHeader section:\n{header_text}"
    return llm.ask_structured(Header, HEADER_SYSTEM, user)


def extract_chunk(rows: list[TableRow], index: int, count: int, header: Header, columns: list[str], overlap: int) -> BaseModel:
    context = f"Document context: Vendor = {header.vendor_name}, Currency = {header.currency}, Invoice = {header.invoice_number}"
    position = f"Chunk {index + 1} of {count} (rows {rows[0].id}-{rows[-1].id})"
    if index > 0 and overlap:
        position += f"; the first {overlap} rows overlap the previous chunk"
    body = "\n".join(f"{r.id} | {r.text}" for r in rows)
    user = f"{context}\nAfter the row id, the cells are in this order: {' | '.join(columns)}\n{position}.\n\n{body}"
    return llm.ask_structured(chunk_schema(tuple(columns)), CHUNK_SYSTEM, user)


def extract_full(text: str) -> FullInvoice:
    return llm.ask_structured(FullInvoice, GENERIC_SYSTEM, text)
