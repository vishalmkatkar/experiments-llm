# Self-correcting RAG with LangGraph

A runnable version of the architecture in
[blog/self-correcting-rag-in-production.html](../../../blog/self-correcting-rag-in-production.html),
using local Ollama models (`qwen2.5:7b` for chat, `nomic-embed-text` for embeddings)
and a small sample knowledge base for a fictional company.

## Run

```bash
cd src
../.venv/bin/python -m app.self_correcting_rag.main                     # all 12 demos
../.venv/bin/python -m app.self_correcting_rag.main --only 2,5,10       # selected demos
../.venv/bin/python -m app.self_correcting_rag.main -q "How much PTO do I get?" -g engineering
../.venv/bin/python -m app.self_correcting_rag.main --mermaid graph.mmd # export the graph
```

Override the models with `RAG_CHAT_MODEL`, `RAG_EMBED_MODEL` and `OLLAMA_HOST`.

## Stage → node map

| Blog stage | Graph node(s) | Notes |
|---|---|---|
| 0 Request setup | `setup` | Starts the request budget (`config.BUDGET`) and trace |
| 1 Input guardrails ∥ router | `input_guard`, `router` → `gate` | Run in parallel, joined at `gate` |
| 2 Cache + query understanding | `cache_lookup`, `rewrite_query` | Cache key includes user groups and index version; rewrite only calls the LLM when there is history |
| 3 Hybrid retrieval + rerank | `retrieve` | BM25 + dense with RRF; permission filter applied inside retrieval |
| 4 Evidence grading | `grade_evidence`, `corrective_rewrite` | Score bands; LLM grader only for the ambiguous band; at most one corrective retrieval |
| 5 Generation with citations | `generate` | Retrieved text in `<doc>` blocks; a citation on every factual sentence |
| 6 Grounding verification | `verify` | Strip ≤20% unsupported claims, otherwise regenerate once, then partial answer or handoff |
| 7 Output guardrails | `output_guard` | PII redaction and a permission check on citations |
| — Exit | `refuse`, `decline_scope`, `not_found`, `finalize` | Every branch ends at `finalize`, which also writes the cache |

Ingestion also removes sentences that look like instructions to a model
(`guards.sanitize_document`), and logs what it removed.

## Demos and the paths they exercise

| # | Scenario | Expected path |
|---|---|---|
| 1 | Small talk | router → direct answer |
| 2 | Parental leave | standard RAG, grounded |
| 3 | Same question again | cache hit |
| 4 | Follow-up "And if we adopt instead?" | rewrite with history |
| 5 | Stipend + standing desk | unsupported claim caught → regenerate → partial answer with caveat |
| 6 | "Vacation days" (document says PTO) | ambiguous band → LLM grader keeps it |
| 7 | Dog in the office | best score far below threshold → not found, no retry |
| 8 / 9 | Q3 revenue target as engineering / finance | permission filter: not found / answered |
| 10 | Kitchen rules (document had a planted instruction) | removed at ingestion, clean answer |
| 11 | Injection in the question | refused by input guardrail |
| 12 | Poem request | declined as out of scope |

A 7B model is not fully deterministic, so individual paths can vary between runs.

## What to swap for production

- `index.HybridIndex.rerank`: replace cosine similarity with a cross-encoder, then recalibrate `TAU_*`.
- `verify`: replace the LLM fact-check with an NLI model, so the checker doesn't share the generator's errors.
- `guards`: replace the regex rules with trained classifiers.
- `Pipeline.cache`: replace the in-process dict with Redis or similar.
- Add tracing (LangSmith or OpenTelemetry). The `trace` field shows which spans to emit.
- Streaming: not implemented here. See the streaming section of the blog post.
