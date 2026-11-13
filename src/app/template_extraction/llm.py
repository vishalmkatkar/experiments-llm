"""Thin wrapper around ChatOllama for structured output."""

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from . import config

_llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0, num_ctx=config.NUM_CTX)

T = TypeVar("T", bound=BaseModel)


def ask_structured(schema: type[T], system: str, user: str) -> T:
    out = _llm.with_structured_output(schema, include_raw=True).invoke([SystemMessage(system), HumanMessage(user)])
    if out["parsed"] is None:
        raise ValueError(f"Model output did not match {schema.__name__}: {out['parsing_error']}")
    return out["parsed"]
