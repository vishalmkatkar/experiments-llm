"""Graph state.

`llm_calls`, `tokens` and `trace` use additive reducers. That lets the input
guardrails and the router run in parallel and both write to them safely.
"""

import operator
from typing import Annotated, TypedDict


class Evidence(TypedDict):
    id: str
    title: str
    source: str
    text: str
    score: float


class RAGState(TypedDict, total=False):
    # Input
    question: str
    history: list[tuple[str, str]]  # (user, assistant) turns, oldest first
    user_groups: list[str]

    # Request budget
    started_at: float
    llm_calls: Annotated[int, operator.add]
    tokens: Annotated[int, operator.add]
    corrections: int
    retrieval_retries: int
    regenerations: int

    # Observability: one line per step, in order
    trace: Annotated[list[str], operator.add]

    # Stage outputs
    guard_ok: bool
    guard_reason: str
    route: str
    cache_hit: bool
    search_query: str
    hits: list[Evidence]
    evidence: list[Evidence]
    evidence_grade: str  # sufficient | partial | none_near | none_far
    missing: list[str]
    draft: str
    feedback: list[str]
    verify_outcome: str  # grounded | regenerate | partial | handoff | not_covered

    # Output
    answer: str
    status: str  # answered | partial | not_found | refused | declined | handoff
    citations: list[str]
    elapsed_s: float
