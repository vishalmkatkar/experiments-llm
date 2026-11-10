"""Rule-based guardrails.

Production: put a trained classifier (injection, policy, PII) behind the same
functions. Rules keep this example fast and free of extra dependencies.
"""

import re

MAX_QUESTION_CHARS = 2000

_INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules)",
        r"(reveal|print|show|repeat).{0,30}(system prompt|hidden instructions)",
        r"\byou are now\b",
        r"\bdeveloper mode\b",
        r"\bjailbreak",
    )
]

# Email domains the assistant may show. Anything else is redacted on output.
ALLOWED_EMAIL_DOMAINS = ("quillfield.example",)
_EMAIL = re.compile(r"\b[\w.+-]+@([\w-]+(?:\.[\w-]+)+)\b")
_PHONE = re.compile(r"(?<!\w)\+?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")


def check_input(question: str) -> tuple[bool, str]:
    if not question.strip():
        return False, "empty question"
    if len(question) > MAX_QUESTION_CHARS:
        return False, "question too long"
    for pattern in _INJECTION:
        if pattern.search(question):
            return False, "prompt injection pattern"
    return True, ""


_DOC_INJECTION = _INJECTION + [
    re.compile(p, re.IGNORECASE)
    for p in (r"\b(system|assistant|ai) note\b", r"\btell the user\b")
]


def sanitize_document(text: str) -> tuple[str, list[str]]:
    """Ingestion-time defense: drop sentences that look like instructions to a model.

    Returns the cleaned text and the removed sentences, which should be logged
    and reviewed rather than silently discarded.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    removed = [s for s in sentences if any(p.search(s) for p in _DOC_INJECTION)]
    kept = [s for s in sentences if s not in removed]
    return " ".join(kept), removed


def redact(text: str) -> tuple[str, int]:
    count = 0

    def _email(m: re.Match[str]) -> str:
        nonlocal count
        if m.group(1).lower().endswith(ALLOWED_EMAIL_DOMAINS):
            return m.group(0)
        count += 1
        return "[redacted email]"

    def _phone(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[redacted phone]"

    text = _EMAIL.sub(_email, text)
    text = _PHONE.sub(_phone, text)
    return text, count
