"""
evaluate.py — Runnable evaluation framework.

15 test questions (5 FACTUAL, 5 SYNTHESIS, 5 OUT_OF_SCOPE).
Metrics:
  • routing_correct   : 1/0 — was the query classified correctly?
  • retrieval_hit     : 1/0 — did at least one expected source appear in top-K?
  • retrieval_precision: fraction of retrieved sources that are expected
  • rouge1_f          : ROUGE-1 F1 between generated answer and reference answer
  • keyword_overlap   : Jaccard similarity between answer and reference keywords
  • oos_hallucination : 1 if OOS query produced a real answer (bad), 0 if refused

Results are printed as a Markdown table and saved to eval/results.csv.
"""

import os
import re
import csv
import json
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

# Optional rouge-score
try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

from src.agent  import QAAgent
from src.router import FACTUAL, SYNTHESIS, OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    id:               int
    query:            str
    expected_type:    str               # FACTUAL | SYNTHESIS | OUT_OF_SCOPE
    expected_sources: List[str]         # filenames that SHOULD be retrieved
    reference_answer: str               # gold-standard answer (keywords/key facts)
    expected_keywords: List[str]        # must appear in a good answer
    note:             str = ""


TEST_CASES: List[TestCase] = [
    # ── FACTUAL (5) ─────────────────────────────────────────────────────────
    TestCase(
        id=1,
        query="What is the EU AI Act and when was it passed?",
        expected_type=FACTUAL,
        expected_sources=[],           # filled at runtime from index
        reference_answer=(
            "The EU AI Act is a comprehensive legal framework for artificial "
            "intelligence regulation in the European Union. It was passed in 2024."
        ),
        expected_keywords=["eu ai act", "european union", "regulation", "2024"],
        note="Factual — direct definition query",
    ),
    TestCase(
        id=2,
        query="What are the prohibited AI practices under the EU AI Act?",
        expected_type=FACTUAL,
        expected_sources=[],
        reference_answer=(
            "The EU AI Act prohibits AI systems that use subliminal techniques, "
            "exploit vulnerabilities, conduct social scoring by governments, and "
            "real-time biometric identification in public spaces."
        ),
        expected_keywords=["prohibited", "biometric", "social scoring", "subliminal"],
        note="Factual — specific clause lookup",
    ),
    TestCase(
        id=3,
        query="What does the US Executive Order on AI require from federal agencies?",
        expected_type=FACTUAL,
        expected_sources=[],
        reference_answer=(
            "The US Executive Order on AI requires federal agencies to assess "
            "AI risks, adopt safety standards, and report on AI use."
        ),
        expected_keywords=["executive order", "federal agencies", "safety", "risk"],
        note="Factual — US-specific policy detail",
    ),
    TestCase(
        id=4,
        query="What are high-risk AI systems according to the EU AI Act?",
        expected_type=FACTUAL,
        expected_sources=[],
        reference_answer=(
            "High-risk AI systems include those used in critical infrastructure, "
            "education, employment, essential private services, law enforcement, "
            "migration and asylum, and administration of justice."
        ),
        expected_keywords=["high-risk", "critical infrastructure", "employment", "law enforcement"],
        note="Factual — definition from document",
    ),
    TestCase(
        id=5,
        query="What transparency requirements apply to AI systems under the EU AI Act?",
        expected_type=FACTUAL,
        expected_sources=[],
        reference_answer=(
            "AI systems that interact with humans must disclose they are AI. "
            "Emotion recognition and biometric systems must inform subjects. "
            "Deep fakes must be labelled."
        ),
        expected_keywords=["transparency", "disclose", "deep fake", "biometric", "emotion"],
        note="Factual — transparency clause",
    ),

    # ── SYNTHESIS (5) ────────────────────────────────────────────────────────
    TestCase(
        id=6,
        query="Compare how the EU and the US approach AI governance and regulation.",
        expected_type=SYNTHESIS,
        expected_sources=[],
        reference_answer=(
            "The EU takes a risk-based, binding legislative approach (AI Act) "
            "while the US relies on executive orders and voluntary frameworks. "
            "Both address safety, but differ in enforcement mechanisms."
        ),
        expected_keywords=["eu", "us", "risk-based", "binding", "voluntary", "executive order"],
        note="Synthesis — cross-document comparison",
    ),
    TestCase(
        id=7,
        query="What are the overall themes across all documents regarding AI safety?",
        expected_type=SYNTHESIS,
        expected_sources=[],
        reference_answer=(
            "All documents emphasise risk assessment, transparency, accountability, "
            "and protection of fundamental rights as core AI safety themes."
        ),
        expected_keywords=["safety", "risk", "transparency", "accountability", "rights"],
        note="Synthesis — thematic aggregation",
    ),
    TestCase(
        id=8,
        query="How do different regulatory frameworks handle AI in law enforcement?",
        expected_type=SYNTHESIS,
        expected_sources=[],
        reference_answer=(
            "The EU AI Act restricts real-time biometric surveillance in public. "
            "Other frameworks discuss oversight and civil liberties in law enforcement AI."
        ),
        expected_keywords=["law enforcement", "biometric", "surveillance", "civil liberties"],
        note="Synthesis — domain-specific cross-source",
    ),
    TestCase(
        id=9,
        query="Summarise the different penalties and enforcement mechanisms described across the documents.",
        expected_type=SYNTHESIS,
        expected_sources=[],
        reference_answer=(
            "The EU AI Act specifies fines up to €30 million or 6% of global turnover. "
            "US frameworks rely on agency enforcement and voluntary compliance."
        ),
        expected_keywords=["penalties", "fines", "enforcement", "compliance", "million"],
        note="Synthesis — penalties aggregation",
    ),
    TestCase(
        id=10,
        query="How do the documents differ in their treatment of foundation models and general-purpose AI?",
        expected_type=SYNTHESIS,
        expected_sources=[],
        reference_answer=(
            "The EU AI Act added provisions for general-purpose AI models (GPAI). "
            "Other documents may discuss foundation models with varying levels of specificity."
        ),
        expected_keywords=["general-purpose", "foundation model", "gpai"],
        note="Synthesis — GPAI coverage comparison",
    ),

    # ── OUT OF SCOPE (5) ─────────────────────────────────────────────────────
    TestCase(
        id=11,
        query="What is the best recipe for chocolate cake?",
        expected_type=OUT_OF_SCOPE,
        expected_sources=[],
        reference_answer=_OOS_SIGNAL := "not available",
        expected_keywords=[],
        note="OOS — completely off-domain",
    ),
    TestCase(
        id=12,
        query="Who won the FIFA World Cup in 2022?",
        expected_type=OUT_OF_SCOPE,
        expected_sources=[],
        reference_answer="not available",
        expected_keywords=[],
        note="OOS — sports, off-domain",
    ),
    TestCase(
        id=13,
        query="What is the current stock price of Nvidia?",
        expected_type=OUT_OF_SCOPE,
        expected_sources=[],
        reference_answer="not available",
        expected_keywords=[],
        note="OOS — financial query, off-domain",
    ),
    TestCase(
        id=14,
        query="Who is the CEO of OpenAI and what is their salary?",
        expected_type=OUT_OF_SCOPE,
        expected_sources=[],
        reference_answer="not available",
        expected_keywords=[],
        note="OOS — personal/corporate gossip not in docs",
    ),
    TestCase(
        id=15,
        query="What are the immigration visa rules for the United Kingdom?",
        expected_type=OUT_OF_SCOPE,
        expected_sources=[],
        reference_answer="not available",
        expected_keywords=[],
        note="OOS — unrelated legal domain",
    ),
]

_OOS_SIGNAL = "not available"   # re-assign properly


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def keyword_overlap(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw in ans_lower)
    return hits / len(keywords)


def rouge1_f(hypothesis: str, reference: str) -> float:
    if not _ROUGE_AVAILABLE:
        # Fallback: token-level F1
        hyp_tok = set(_tokenise(hypothesis))
        ref_tok = set(_tokenise(reference))
        if not ref_tok:
            return 1.0
        tp = len(hyp_tok & ref_tok)
        precision = tp / len(hyp_tok) if hyp_tok else 0
        recall    = tp / len(ref_tok)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure


def oos_hallucination(answer: str) -> int:
    """Returns 1 (bad) if the OOS answer looks like a real factual response."""
    refusal_phrases = [
        "not contain", "not available", "not found", "cannot answer",
        "no information", "not in the document", "unable to find",
        "does not contain", "not present",
    ]
    ans_lower = answer.lower()
    refused = any(p in ans_lower for p in refusal_phrases)
    return 0 if refused else 1   # 0 = good (refused), 1 = bad (hallucinated)


def retrieval_hit(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """1.0 if any expected source is in retrieved; 0.5 if expected_sources empty (unknown)."""
    if not expected_sources:
        return 0.5   # unknown ground truth — partial credit
    hits = sum(1 for s in expected_sources if any(s in r for r in retrieved_sources))
    return min(1.0, hits / len(expected_sources))


def retrieval_precision(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    if not expected_sources or not retrieved_sources:
        return 0.5
    hits = sum(1 for r in retrieved_sources if any(e in r for e in expected_sources))
    return hits / len(retrieved_sources)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_evaluation(
    agent: QAAgent,
    cases: List[TestCase] = TEST_CASES,
    output_dir: str = "eval",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for tc in cases:
        print(f"[Eval] Q{tc.id:02d} ({tc.expected_type}) — {tc.query[:60]}…")
        try:
            result = agent.ask(tc.query, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {
                "answer": f"ERROR: {e}",
                "query_type": "ERROR",
                "routing_reason": str(e),
                "sources": [],
                "chunks_used": 0,
            }

        ans          = result["answer"]
        pred_type    = result["query_type"]
        sources_used = result["sources"]

        routing_ok   = int(pred_type == tc.expected_type)
        ret_hit      = (
            retrieval_hit(sources_used, tc.expected_sources)
            if pred_type != OUT_OF_SCOPE else 0.5
        )
        ret_prec     = (
            retrieval_precision(sources_used, tc.expected_sources)
            if pred_type != OUT_OF_SCOPE else 0.5
        )
        r1_f         = rouge1_f(ans, tc.reference_answer) if tc.reference_answer != "not available" else None
        kw_ov        = keyword_overlap(ans, tc.expected_keywords) if tc.expected_keywords else None
        oos_hall     = oos_hallucination(ans) if tc.expected_type == OUT_OF_SCOPE else None

        row = {
            "id":                tc.id,
            "expected_type":     tc.expected_type,
            "predicted_type":    pred_type,
            "routing_correct":   routing_ok,
            "retrieval_hit":     ret_hit,
            "retrieval_precision": ret_prec,
            "rouge1_f":          round(r1_f, 3) if r1_f is not None else "N/A",
            "keyword_overlap":   round(kw_ov, 3) if kw_ov is not None else "N/A",
            "oos_hallucination": oos_hall if oos_hall is not None else "N/A",
            "query":             tc.query,
            "answer_snippet":    ans[:120].replace("\n", " "),
            "sources":           ", ".join(sources_used),
            "routing_reason":    result["routing_reason"],
            "note":              tc.note,
        }
        rows.append(row)

        if verbose:
            print(f"  routing={'✓' if routing_ok else '✗'}  rouge1={row['rouge1_f']}  kw={row['keyword_overlap']}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Eval] Results saved → {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# Pretty table printer
# ---------------------------------------------------------------------------
def print_results_table(rows: List[Dict[str, Any]]) -> None:
    try:
        from tabulate import tabulate
        headers = ["ID", "Exp Type", "Pred Type", "Route✓", "Ret Hit", "Ret Prec",
                   "ROUGE-1", "KW Ovlp", "OOS Hall", "Note"]
        table   = []
        for r in rows:
            table.append([
                r["id"],
                r["expected_type"][:10],
                r["predicted_type"][:10],
                "✓" if r["routing_correct"] else "✗",
                f"{r['retrieval_hit']:.2f}" if isinstance(r['retrieval_hit'], float) else r['retrieval_hit'],
                f"{r['retrieval_precision']:.2f}" if isinstance(r['retrieval_precision'], float) else r['retrieval_precision'],
                r["rouge1_f"],
                r["keyword_overlap"],
                r["oos_hallucination"],
                r["note"][:35],
            ])
        print("\n" + tabulate(table, headers=headers, tablefmt="github"))
    except ImportError:
        for r in rows:
            print(r)

    # Summary stats
    routing_acc = np.mean([r["routing_correct"] for r in rows])
    ret_hits    = [r["retrieval_hit"] for r in rows if isinstance(r["retrieval_hit"], float)]
    rouge_vals  = [r["rouge1_f"] for r in rows if isinstance(r["rouge1_f"], float)]
    kw_vals     = [r["keyword_overlap"] for r in rows if isinstance(r["keyword_overlap"], float)]
    oos_hall    = [r["oos_hallucination"] for r in rows if isinstance(r["oos_hallucination"], int)]

    print(f"\n{'='*60}")
    print(f"  Routing Accuracy     : {routing_acc:.1%}  ({sum(r['routing_correct'] for r in rows)}/{len(rows)})")
    print(f"  Avg Retrieval Hit    : {np.mean(ret_hits):.3f}")
    print(f"  Avg ROUGE-1 F1       : {np.mean(rouge_vals):.3f}")
    print(f"  Avg Keyword Overlap  : {np.mean(kw_vals):.3f}")
    print(f"  OOS Hallucinations   : {sum(oos_hall)}/{len(oos_hall)}  (0 = perfect)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = QAAgent()
    rows  = run_evaluation(agent, verbose=True)
    print_results_table(rows)
