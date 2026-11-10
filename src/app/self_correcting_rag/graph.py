"""Wires the pipeline's nodes into a LangGraph StateGraph."""

from langgraph.graph import END, START, StateGraph

from .corpus import CORPUS
from .index import HybridIndex
from .nodes import Pipeline
from .state import RAGState

NODES = [
    "setup", "input_guard", "router", "gate", "refuse", "decline_scope", "direct_answer",
    "cache_lookup", "rewrite_query", "retrieve", "grade_evidence", "corrective_rewrite",
    "not_found", "generate", "verify", "output_guard", "finalize",
]


def build_graph(pipeline: Pipeline):
    g = StateGraph(RAGState)
    for name in NODES:
        g.add_node(name, getattr(pipeline, name))

    # 0 -> 1: guardrails and router run in parallel, then join at the gate.
    g.add_edge(START, "setup")
    g.add_edge("setup", "input_guard")
    g.add_edge("setup", "router")
    g.add_edge(["input_guard", "router"], "gate")
    g.add_conditional_edges("gate", pipeline.after_gate, ["refuse", "direct_answer", "decline_scope", "cache_lookup"])

    # 2 -> 4: cache, query understanding, retrieval, grading.
    g.add_conditional_edges("cache_lookup", pipeline.after_cache, ["output_guard", "rewrite_query"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "grade_evidence")
    g.add_conditional_edges("grade_evidence", pipeline.after_grade, ["generate", "corrective_rewrite", "not_found"])
    g.add_edge("corrective_rewrite", "retrieve")

    # 5 -> 6: generation and grounding, with at most one regeneration.
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", pipeline.after_verify, ["generate", "output_guard", "finalize"])

    # 7 -> end: every branch ends at finalize.
    g.add_edge("direct_answer", "output_guard")
    g.add_edge("output_guard", "finalize")
    for terminal in ("refuse", "decline_scope", "not_found"):
        g.add_edge(terminal, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def create_app():
    """Builds the index (embeds the corpus) and returns (compiled_graph, pipeline)."""
    pipeline = Pipeline(HybridIndex(CORPUS))
    return build_graph(pipeline), pipeline


def ask(app, question: str, groups: list[str], history: list[tuple[str, str]] | None = None) -> dict:
    return app.invoke(
        {"question": question, "user_groups": groups, "history": history or [], "llm_calls": 0, "tokens": 0, "trace": []},
        config={"recursion_limit": 40},
    )
