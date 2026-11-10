"""Tunable settings for the self-correcting RAG example.

Every threshold here is a starting point. Calibrate them on a labelled
question set that includes questions the system should decline.
"""

import os
from dataclasses import dataclass

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass(frozen=True)
class Budget:
    """One budget per request. Every loop in the graph draws from it."""

    max_llm_calls: int = 8
    deadline_s: float = 120.0  # generous because a local 7B model is slow
    max_corrections: int = 2  # corrective retrievals + regenerations, combined
    max_retrieval_retries: int = 1
    max_regenerations: int = 1


BUDGET = Budget()

# Rerank score bands. Scores are cosine similarities from nomic-embed-text,
# standing in for a cross-encoder (see index.HybridIndex.rerank).
TAU_HIGH = 0.76  # at or above: relevant, no LLM grader needed
TAU_LOW = 0.65  # below: irrelevant, no LLM grader needed
NONE_FAR = 0.60  # best score below this: answer is almost certainly not in the corpus

# Grounding: if at most this share of claims is unsupported, strip them
# instead of regenerating.
STRIP_MAX = 0.20

# Retrieval
TOP_K_EACH = 6  # candidates from each retriever before fusion
FINAL_K = 4  # chunks kept after rerank
RRF_K = 60  # reciprocal rank fusion constant
