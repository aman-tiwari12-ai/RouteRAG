"""
router.py — Explicit, inspectable query router.

The router uses a TWO-STAGE decision pipeline:

  Stage 1 — Lexical pre-filter (rule-based, always runs first)
    Checks for hard signals that the query is out of scope:
    • Query contains topic keywords known to be absent from the document set
      (sports, recipes, weather, personal finance, etc.)
    • Query is a greeting / small talk with no question content

  Stage 2 — Retrieval-confidence gate (runs after vector search)
    After retrieving the top-K chunks, we score retrieval confidence:
    • max_sim  : cosine similarity of the best chunk
    • avg_sim  : mean of top-K similarities
    • coverage : fraction of query content words found in retrieved chunks

    Decision tree:
      if max_sim < SIM_THRESHOLD_LOW                 → OUT_OF_SCOPE
      elif max_sim ≥ SIM_THRESHOLD_HIGH
           and n_sources == 1                        → FACTUAL
      elif max_sim ≥ SIM_THRESHOLD_LOW
           and (n_sources > 1 OR synthesis_signal)  → SYNTHESIS
      else                                           → OUT_OF_SCOPE

    synthesis_signal: the query contains words like "compare", "across",
    "difference", "both", "all", "how do", "why", "overall", "summarise".

This design makes routing fully transparent—every decision is traceable to
concrete numeric thresholds and keyword lists, with no black-box LLM routing.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Thresholds — tweak here if precision/recall trade-offs need adjustment
# ---------------------------------------------------------------------------
SIM_THRESHOLD_LOW  = 0.30   # below this → definitely out of scope
SIM_THRESHOLD_HIGH = 0.52   # above this (single source) → factual

# Minimum content-word coverage for a retrieved chunk to count as "on topic"
COVERAGE_THRESHOLD = 0.25

# Query types
FACTUAL      = "FACTUAL"
SYNTHESIS    = "SYNTHESIS"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------
_OOS_TOPICS = {
    # completely off-domain signals
    "recipe", "cooking", "sport", "football", "cricket", "movie", "film",
    "stock", "invest", "weather", "forecast", "celebrity", "gossip",
    "astrology", "horoscope", "game", "music", "song", "lyric",
}

_SYNTHESIS_SIGNALS = {
    "compare", "comparison", "contrast", "difference", "differ",
    "across", "between", "both", "all", "multiple", "various",
    "summarise", "summarize", "overall", "generally", "broadly",
    "how do", "why do", "what are the main", "outline", "overview",
    "perspectives", "views", "approaches", "frameworks",
}

_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s*(morning|evening|afternoon)|"
    r"how are you|what'?s up|sup)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RoutingDecision:
    query_type:    str                     # FACTUAL | SYNTHESIS | OUT_OF_SCOPE
    max_sim:       float = 0.0
    avg_sim:       float = 0.0
    coverage:      float = 0.0
    n_sources:     int   = 0
    synthesis_hit: bool  = False
    oos_rule:      str   = ""              # which rule fired for OOS
    chunks:        List[Dict[str, Any]] = field(default_factory=list)

    def explain(self) -> str:
        if self.query_type == OUT_OF_SCOPE:
            return (
                f"OUT_OF_SCOPE — {self.oos_rule} "
                f"(max_sim={self.max_sim:.3f}, coverage={self.coverage:.3f})"
            )
        if self.query_type == FACTUAL:
            return (
                f"FACTUAL — max_sim={self.max_sim:.3f}, "
                f"n_sources={self.n_sources}, synthesis_signal=False"
            )
        return (
            f"SYNTHESIS — max_sim={self.max_sim:.3f}, "
            f"n_sources={self.n_sources}, synthesis_signal={self.synthesis_hit}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stopwords() -> set:
    try:
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except Exception:
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "do", "does", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "with", "that", "this", "it", "its",
            "what", "how", "why", "when", "which", "who",
        }

_SW = None
def stopwords() -> set:
    global _SW
    if _SW is None:
        _SW = _stopwords()
    return _SW


def content_words(text: str) -> List[str]:
    """Return lowercase, non-stop, alphabetic tokens."""
    sw = stopwords()
    return [
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in sw and len(w) > 2
    ]


def coverage_score(query: str, chunks: List[Dict[str, Any]]) -> float:
    """Fraction of query content words found in the union of chunk texts."""
    q_words = set(content_words(query))
    if not q_words:
        return 0.0
    chunk_union = " ".join(c['text'] for c in chunks).lower()
    found = sum(1 for w in q_words if w in chunk_union)
    return found / len(q_words)


def has_synthesis_signal(query: str) -> bool:
    q_lower = query.lower()
    return any(sig in q_lower for sig in _SYNTHESIS_SIGNALS)


def has_oos_topic(query: str) -> bool:
    words = set(re.findall(r"[a-z]+", query.lower()))
    return bool(words & _OOS_TOPICS)


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
class QueryRouter:
    """
    Stateless router.  Call route(query, retrieved_chunks) after vector search.
    """

    def route(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> RoutingDecision:
        # ── Stage 1: Lexical pre-filter ──────────────────────────────────────
        if _GREETING_PATTERNS.match(query):
            return RoutingDecision(
                query_type=OUT_OF_SCOPE,
                oos_rule="greeting/small-talk detected",
            )

        if has_oos_topic(query):
            return RoutingDecision(
                query_type=OUT_OF_SCOPE,
                oos_rule="out-of-domain topic keyword matched",
            )

        # ── Stage 2: Retrieval-confidence gate ───────────────────────────────
        if not retrieved_chunks:
            return RoutingDecision(
                query_type=OUT_OF_SCOPE,
                oos_rule="no chunks retrieved",
            )

        sims      = [c.get('similarity', 0.0) for c in retrieved_chunks]
        max_sim   = max(sims)
        avg_sim   = sum(sims) / len(sims)
        coverage  = coverage_score(query, retrieved_chunks)
        n_sources = len({c['source'] for c in retrieved_chunks})
        syn_hit   = has_synthesis_signal(query)

        if max_sim < SIM_THRESHOLD_LOW or coverage < COVERAGE_THRESHOLD:
            return RoutingDecision(
                query_type=OUT_OF_SCOPE,
                max_sim=max_sim,
                avg_sim=avg_sim,
                coverage=coverage,
                n_sources=n_sources,
                synthesis_hit=syn_hit,
                oos_rule=(
                    f"max_sim={max_sim:.3f} < {SIM_THRESHOLD_LOW} "
                    f"OR coverage={coverage:.3f} < {COVERAGE_THRESHOLD}"
                ),
                chunks=retrieved_chunks,
            )

        # High-confidence single-source factual
        if max_sim >= SIM_THRESHOLD_HIGH and not syn_hit and n_sources == 1:
            return RoutingDecision(
                query_type=FACTUAL,
                max_sim=max_sim,
                avg_sim=avg_sim,
                coverage=coverage,
                n_sources=n_sources,
                synthesis_hit=False,
                chunks=retrieved_chunks,
            )

        # Multi-source or explicit synthesis cue
        if n_sources > 1 or syn_hit:
            return RoutingDecision(
                query_type=SYNTHESIS,
                max_sim=max_sim,
                avg_sim=avg_sim,
                coverage=coverage,
                n_sources=n_sources,
                synthesis_hit=syn_hit,
                chunks=retrieved_chunks,
            )

        # Single source but mid-range similarity — treat as factual
        if max_sim >= SIM_THRESHOLD_LOW:
            return RoutingDecision(
                query_type=FACTUAL,
                max_sim=max_sim,
                avg_sim=avg_sim,
                coverage=coverage,
                n_sources=n_sources,
                synthesis_hit=False,
                chunks=retrieved_chunks,
            )

        return RoutingDecision(
            query_type=OUT_OF_SCOPE,
            max_sim=max_sim,
            avg_sim=avg_sim,
            coverage=coverage,
            oos_rule="fallback — no routing rule matched",
            chunks=retrieved_chunks,
        )


if __name__ == "__main__":
    # Quick self-test
    router = QueryRouter()
    test_cases = [
        ("What are the penalties for GDPR violations?",
         [{"source": "gdpr.txt", "similarity": 0.72, "text": "gdpr violations penalties fines"}]),
        ("Compare the EU AI Act and the US AI Bill of Rights",
         [{"source": "eu_ai.txt", "similarity": 0.61, "text": "eu ai act regulation"},
          {"source": "us_ai.txt", "similarity": 0.58, "text": "us ai bill rights"}]),
        ("What is the best cricket team?",
         [{"source": "eu_ai.txt", "similarity": 0.12, "text": "eu ai act"}]),
    ]
    for q, chunks in test_cases:
        d = router.route(q, chunks)
        print(f"Q: {q}\n  → {d.explain()}\n")
