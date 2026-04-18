"""
failure_analysis.py — Documents ≥3 known failure modes with root causes.

Outputs a structured report and a per-failure diagnostic table.
This is executable code — run it directly to produce the report.
"""

import textwrap
from typing import List, Dict


FAILURES: List[Dict] = [
    {
        "id": 1,
        "title": "Mid-similarity FACTUAL mis-routed as OUT_OF_SCOPE",
        "description": textwrap.dedent("""
            Queries whose answers exist in documents but use vocabulary different from
            the document text produce cosine similarities in the 0.28–0.35 range —
            just below SIM_THRESHOLD_LOW (0.30).  The router therefore fires the
            OOS path even though a correct answer is available.

            Example: "What safeguards exist for vulnerable populations in AI systems?"
            The documents discuss "children", "elderly", "persons with disabilities"
            but the query uses "vulnerable populations" — a paraphrase that reduces
            similarity below the threshold.
        """).strip(),
        "root_cause": textwrap.dedent("""
            all-MiniLM-L6-v2 is a compact 384-dim model optimised for general
            semantic similarity.  Domain-specific regulatory paraphrases (GDPR ↔ data
            protection, vulnerable populations ↔ children / elderly) produce lower
            cosine scores than ideal.  The fixed SIM_THRESHOLD_LOW is too aggressive.
        """).strip(),
        "mitigation": textwrap.dedent("""
            1. Use a domain-adapted model (e.g. legal-bert, InstructorXL with
               instruction prefix "Represent the AI regulation query:").
            2. Lower SIM_THRESHOLD_LOW to 0.25 and compensate with a stricter
               keyword coverage check (COVERAGE_THRESHOLD 0.30 → 0.35).
            3. Add query expansion: generate 2–3 synonym phrases for the query
               using a lightweight LLM call, embed all, take the max similarity.
        """).strip(),
        "query_example": "What safeguards exist for vulnerable populations?",
        "expected": FACTUAL  := "FACTUAL",
        "predicted": OUT_OF_SCOPE := "OUT_OF_SCOPE",
    },
    {
        "id": 2,
        "title": "SYNTHESIS query mis-classified as FACTUAL for single-source clusters",
        "description": textwrap.dedent("""
            When a synthesis query (e.g. "Compare EU and US enforcement") retrieves
            chunks that happen to cluster around one document (because one document
            is more verbose on the topic), n_sources == 1 and synthesis_signal may
            not fire if the query phrasing is neutral ("What are the enforcement
            mechanisms?").  The router then returns FACTUAL, producing a one-sided
            answer that ignores the other document's perspective.
        """).strip(),
        "root_cause": textwrap.dedent("""
            FAISS top-K retrieval is biased toward the document with the most
            keyword density on the query topic.  If document A has 3 chunks on
            enforcement and document B has 1, all 5 retrieved chunks may come from A.
            The synthesis signal relies on explicit comparative language; neutral
            phrasings bypass it.
        """).strip(),
        "mitigation": textwrap.dedent("""
            1. Implement Maximal Marginal Relevance (MMR) retrieval to diversify
               sources in the top-K result set.
            2. Add a source-diversity check: if only 1 source is returned but the
               index has ≥2 sources relevant (sim > 0.25), promote to SYNTHESIS.
            3. Post-retrieval re-rank: fetch top-20, then select top-5 that maximise
               both similarity AND source diversity.
        """).strip(),
        "query_example": "What are the enforcement mechanisms for AI regulation?",
        "expected": "SYNTHESIS",
        "predicted": "FACTUAL",
    },
    {
        "id": 3,
        "title": "Partial hallucination on FACTUAL queries spanning chunk boundaries",
        "description": textwrap.dedent("""
            Some answers require a two-sentence definition where sentence 1 is in
            chunk N and sentence 2 is in chunk N+1.  With 512-token chunks and
            80-token overlap, short definitional clauses at chunk end are sometimes
            truncated.  The generator then produces a plausible-sounding completion
            that extends beyond the evidence — a soft hallucination.

            Example: Fines under the EU AI Act.  Chunk 4 contains "up to €30 million"
            but the "or 6% of global annual turnover" clause is at the start of
            chunk 5 — which may not be in the top-K if chunk 5's overall similarity
            is lower due to boilerplate surrounding text.
        """).strip(),
        "root_cause": textwrap.dedent("""
            Fixed-size chunking is semantically unaware.  Regulatory texts often
            have long numbered-list clauses that span paragraph boundaries, making
            80-token overlap insufficient.  The generator's system prompt instructs
            strict grounding but Claude may infer the missing clause from prior
            knowledge when the context is nearly complete.
        """).strip(),
        "mitigation": textwrap.dedent("""
            1. Switch to sentence-aware chunking (spaCy sentence splitter) and set
               overlap to cover a minimum of 2 full sentences.
            2. After retrieval, always fetch the immediately adjacent chunks (N-1, N+1)
               for the highest-similarity chunk — a "neighbour expansion" step.
            3. Strengthen the system prompt: add "If the information is incomplete,
               say 'The document excerpt is incomplete on this point' rather than
               inferring the missing detail."
        """).strip(),
        "query_example": "What are the exact financial penalties under the EU AI Act?",
        "expected": "FACTUAL (complete answer)",
        "predicted": "FACTUAL (partial hallucination on fine amount)",
    },
    {
        "id": 4,
        "title": "OOS queries with AI-adjacent keywords mis-classified as FACTUAL",
        "description": textwrap.dedent("""
            Queries like "Who is Sam Altman and what is his view on AI regulation?"
            share high semantic similarity with regulation document chunks because
            they mention AI regulation — but the person-specific information is
            absent from the documents.  Similarity scores can reach 0.45+, causing
            the router to fire FACTUAL, and the generator produces an answer that
            may blend document content with hallucinated biographical details.
        """).strip(),
        "root_cause": textwrap.dedent("""
            Semantic similarity is topic-level, not entity-level.  The embedding
            model captures "AI regulation" as the dominant signal and ignores that
            "Sam Altman" is a named entity not present in any chunk.
        """).strip(),
        "mitigation": textwrap.dedent("""
            1. Add a Named Entity Recognition (NER) pre-filter: if the query contains
               PER (person) or ORG entities not found in the chunk corpus, increase
               OOS probability.
            2. Implement entity grounding check: after retrieval, verify that key
               nouns from the query appear verbatim in the top-3 chunks.  If not,
               route to OOS.
            3. Expand _OOS_TOPICS with person-query patterns (who is, what did X say).
        """).strip(),
        "query_example": "Who is Sam Altman and what is his view on AI safety?",
        "expected": "OUT_OF_SCOPE",
        "predicted": "FACTUAL (hallucinated answer)",
    },
]


def print_failure_report(failures: List[Dict] = FAILURES) -> None:
    try:
        from tabulate import tabulate
        summary = []
        for f in failures:
            summary.append([
                f["id"],
                f["title"][:45],
                f.get("expected", "—")[:15],
                f.get("predicted", "—")[:15],
                f["root_cause"][:60] + "…",
            ])
        headers = ["#", "Failure Title", "Expected", "Predicted", "Root Cause (truncated)"]
        print("\n" + tabulate(summary, headers=headers, tablefmt="github"))
    except ImportError:
        pass

    for f in failures:
        print(f"\n{'='*70}")
        print(f"FAILURE {f['id']}: {f['title']}")
        print(f"\nDescription:\n  " + f['description'].replace('\n', '\n  '))
        print(f"\nRoot Cause:\n  " + f['root_cause'].replace('\n', '\n  '))
        print(f"\nMitigation:\n  " + f['mitigation'].replace('\n', '\n  '))
        if "query_example" in f:
            print(f"\nExample Query: {f['query_example']}")
        print(f"Expected Routing : {f.get('expected', '—')}")
        print(f"Predicted Routing: {f.get('predicted', '—')}")


if __name__ == "__main__":
    print_failure_report()
