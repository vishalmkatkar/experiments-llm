"""Thin wrappers around ChatOllama that also report token usage for the budget."""

from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from . import config

_llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)

T = TypeVar("T", bound=BaseModel)


def _tokens(msg: BaseMessage | None) -> int:
    usage = getattr(msg, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0))


def ask_text(system: str, user: str) -> tuple[str, int]:
    msg = _llm.invoke([SystemMessage(system), HumanMessage(user)])
    return str(msg.content).strip(), _tokens(msg)


def ask_structured(schema: type[T], system: str, user: str) -> tuple[T, int]:
    out = _llm.with_structured_output(schema, include_raw=True).invoke(
        [SystemMessage(system), HumanMessage(user)]
    )
    parsed = out["parsed"]
    if parsed is None:
        raise ValueError(f"Model output did not match {schema.__name__}: {out['parsing_error']}")
    return parsed, _tokens(out["raw"])
