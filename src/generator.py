"""
generator.py — Grounded answer generation via Anthropic Claude API.

Design principles:
  • All answers are grounded strictly in the retrieved chunks.
  • The system prompt explicitly instructs the model NOT to use outside knowledge.
  • For OUT_OF_SCOPE, no LLM call is made — a deterministic refusal is returned.
  • For FACTUAL, we pass the single best chunk plus its neighbours.
  • For SYNTHESIS, we pass all top-K chunks, deduplicated by chunk ID.
"""

import os
import textwrap
from typing import List, Dict, Any, Optional

from src.router import RoutingDecision, FACTUAL, SYNTHESIS, OUT_OF_SCOPE

# ---------------------------------------------------------------------------
# Anthropic client (lazy import so tests can mock it)
# ---------------------------------------------------------------------------
def _get_client():
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")


MODEL   = "claude-sonnet-4-20250514"
MAX_TOK = 1024

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_SYS_FACTUAL = textwrap.dedent("""
    You are a precise research assistant answering questions about AI regulation.
    Answer ONLY using the provided document excerpts. 
    Do NOT use any outside knowledge.
    If the excerpts do not fully answer the question, say so explicitly.
    Cite the source filename for each claim, e.g. [source: filename.txt].
    Be concise and accurate.
""").strip()

_SYS_SYNTHESIS = textwrap.dedent("""
    You are a research analyst synthesising information from multiple AI-regulation documents.
    Answer ONLY using the provided excerpts — do NOT introduce outside knowledge.
    Explicitly note where sources agree, disagree, or cover different aspects.
    Cite the source filename for each claim, e.g. [source: filename.txt].
    Structure your answer clearly with brief paragraphs.
""").strip()

_OOS_RESPONSE = (
    "The provided documents do not contain sufficient information to answer "
    "this question. Please refer to authoritative external sources."
)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def _build_context(chunks: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    seen_ids  = set()
    excerpts  = []
    total_len = 0
    for chunk in chunks:
        cid = chunk.get('id', chunk['text'][:20])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        excerpt = f"[source: {chunk['source']} | chunk {chunk.get('chunk_idx', '?')}]\n{chunk['text']}"
        if total_len + len(excerpt) > max_chars:
            break
        excerpts.append(excerpt)
        total_len += len(excerpt)
    return "\n\n---\n\n".join(excerpts)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class AnswerGenerator:
    def __init__(self):
        self._client = None   # lazy

    @property
    def client(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    def generate(
        self,
        query: str,
        decision: RoutingDecision,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          answer       : str
          query_type   : str
          routing_reason: str
          sources      : list[str]
          chunks_used  : int
        """
        if decision.query_type == OUT_OF_SCOPE:
            return {
                "answer":         _OOS_RESPONSE,
                "query_type":     OUT_OF_SCOPE,
                "routing_reason": decision.explain(),
                "sources":        [],
                "chunks_used":    0,
            }

        chunks  = decision.chunks
        context = _build_context(chunks)
        sys_prompt = _SYS_FACTUAL if decision.query_type == FACTUAL else _SYS_SYNTHESIS

        user_msg = (
            f"Document excerpts:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
        )

        if verbose:
            print(f"\n[Generator] Type={decision.query_type}, chunks={len(chunks)}")
            print(f"[Generator] Context length: {len(context)} chars")

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOK,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        answer  = response.content[0].text.strip()
        sources = list({c['source'] for c in chunks})

        return {
            "answer":         answer,
            "query_type":     decision.query_type,
            "routing_reason": decision.explain(),
            "sources":        sources,
            "chunks_used":    len(chunks),
        }
