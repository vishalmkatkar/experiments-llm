"""Hybrid retrieval: BM25 + dense embeddings, fused with reciprocal rank fusion.

The permission filter is applied before scoring, inside retrieval, so restricted
chunks never become candidates. Filtering after retrieval is fragile: it can
silently return fewer results and is easy to bypass by accident.
"""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, replace

from langchain_ollama import OllamaEmbeddings

from . import config
from .corpus import Chunk
from .guards import sanitize_document

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from get how i if in is it my of on or "
    "the to we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], i: int) -> float:
        tf, dl = self.tf[i], self.lengths[i]
        total = 0.0
        for t in query:
            f = tf.get(t)
            if f:
                norm = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                total += self.idf[t] * f * (self.k1 + 1) / norm
        return total


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # rerank score, used by the evidence grader's bands


class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        # Ingestion-time injection scan. Removed sentences are kept for review.
        self.quarantined: dict[str, list[str]] = {}
        self.chunks = []
        for c in chunks:
            clean, removed = sanitize_document(c.text)
            if removed:
                self.quarantined[c.id] = removed
                c = replace(c, text=clean)
            self.chunks.append(c)
        self.by_id = {c.id: c for c in self.chunks}
        self.embedder = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
        # nomic-embed-text expects task prefixes on documents and queries.
        self.vectors = self.embedder.embed_documents(
            [f"search_document: {c.title}\n{c.text}" for c in self.chunks]
        )
        self.bm25 = BM25([tokenize(f"{c.title} {c.text}") for c in self.chunks])
        # Part of the cache key, so a re-index invalidates cached answers.
        self.version = hashlib.sha256(
            "".join(c.id + c.text for c in self.chunks).encode()
        ).hexdigest()[:12]

    def can_read(self, chunk_id: str, groups: list[str]) -> bool:
        chunk = self.by_id.get(chunk_id)
        return chunk is not None and ("all" in chunk.groups or bool(chunk.groups & set(groups)))

    def search(self, query: str, groups: list[str]) -> list[Hit]:
        allowed = [i for i, c in enumerate(self.chunks) if self.can_read(c.id, groups)]
        if not allowed:
            return []

        qv = self.embedder.embed_query(f"search_query: {query}")
        dense_scores = {i: _cosine(qv, self.vectors[i]) for i in allowed}
        q_tokens = tokenize(query)
        sparse_scores = {i: self.bm25.score(q_tokens, i) for i in allowed}

        dense = sorted(allowed, key=lambda i: -dense_scores[i])[: config.TOP_K_EACH]
        sparse = [i for i in sorted(allowed, key=lambda i: -sparse_scores[i]) if sparse_scores[i] > 0]
        sparse = sparse[: config.TOP_K_EACH]

        fused: Counter[int] = Counter()
        for ranking in (dense, sparse):
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (config.RRF_K + rank + 1)

        candidates = [i for i, _ in fused.most_common()]
        return self.rerank(candidates, dense_scores)[: config.FINAL_K]

    def rerank(self, candidates: list[int], dense_scores: dict[int, float]) -> list[Hit]:
        """Stand-in for a cross-encoder reranker.

        Production: replace this with a cross-encoder (for example a bge-reranker
        model) scoring (query, chunk) pairs, and recalibrate the TAU bands on its
        score scale. Here, cosine similarity keeps the example dependency-free.
        """
        hits = [Hit(self.chunks[i], dense_scores[i]) for i in candidates]
        return sorted(hits, key=lambda h: -h.score)
