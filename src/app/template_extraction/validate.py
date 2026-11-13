"""Step 4: check the stitched result against the document's own numbers.

An invoice contains its own answer key. These checks don't ask a model how
confident it is; they test the output against arithmetic the document must satisfy.
"""

from dataclasses import asdict, dataclass

from .config import MONEY_TOLERANCE


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= MONEY_TOLERANCE + 1e-9


def run_checks(
    header: dict,
    items: list[dict],
    totals: dict,
    required: list[str],
    expected_rows: int | None,
    counters: list[tuple[int, int] | None] | None,
    overlap_conflicts: list[str],
) -> list[dict]:
    checks: list[Check] = []

    missing = [f for f in required if not (header.get(f) or totals.get(f))]
    checks.append(Check("Required fields", not missing, f"missing {missing}" if missing else "all present"))

    if counters is not None:
        seen = [c for c in counters if c]
        n = seen[0][1] if seen else None
        ok = bool(seen) and len(seen) == len(counters) == n and [c[0] for c in seen] == list(range(1, n + 1))
        checks.append(Check("Page counter", ok, f"pages {[c[0] for c in seen]} of {n}" if seen else "no page counter found"))

    if expected_rows is not None:
        ok = len(items) == expected_rows
        checks.append(Check("Row count", ok, f"{len(items)} extracted vs {expected_rows} detected"))

    bad = [i for i in items if not _close(i["quantity"] * i["unit_price"], i["amount"])]
    checks.append(Check(
        "Line arithmetic", not bad,
        "qty x price = amount on every row" if not bad
        else f"{len(bad)} row(s) fail, e.g. {bad[0].get('row_id', '')} {bad[0]['quantity']} x {bad[0]['unit_price']} != {bad[0]['amount']}",
    ))

    total_lines = round(sum(i["amount"] for i in items), 2)
    sub = totals.get("subtotal")
    checks.append(Check(
        "Table total", sub is not None and _close(total_lines, sub),
        f"sum of lines {total_lines:,.2f} vs subtotal {sub:,.2f}" if sub is not None else "subtotal not found",
    ))

    tax, total = totals.get("tax"), totals.get("total")
    if None not in (sub, tax, total):
        checks.append(Check("Grand total", _close(sub + tax, total), f"{sub:,.2f} + {tax:,.2f} vs {total:,.2f}"))
    else:
        checks.append(Check("Grand total", False, "subtotal, tax or total not found"))

    if overlap_conflicts:
        checks.append(Check("Overlap agreement", False, f"overlapping rows disagree: {overlap_conflicts}"))

    return [asdict(c) for c in checks]
