"""
agent.py — Top-level agent that wires ingestion → retrieval → routing → generation.
"""

import os
from typing import Optional, Dict, Any

from src.ingestion  import VectorStore, ingest_folder
from src.router     import QueryRouter
from src.generator  import AnswerGenerator


class QAAgent:
    def __init__(
        self,
        index_dir: str = "index",
        top_k: int = 5,
        embed_model: Optional[str] = None,
    ):
        self.top_k     = top_k
        self.vs        = VectorStore(*([] if embed_model is None else [embed_model]))
        self.router    = QueryRouter()
        self.generator = AnswerGenerator()
        self._loaded   = False

        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            self.vs.load(index_dir)
            self._loaded = True

    # ------------------------------------------------------------------ #
    def ingest(self, data_dir: str, index_dir: str = "index") -> None:
        """Ingest documents, build FAISS index, save."""
        ingest_folder(data_dir, index_dir)
        self.vs.load(index_dir)
        self._loaded = True

    # ------------------------------------------------------------------ #
    def ask(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """Full pipeline: retrieve → route → generate."""
        if not self._loaded:
            raise RuntimeError("No index loaded. Call ingest() first.")

        # 1. Retrieve
        chunks   = self.vs.search(query, k=self.top_k)

        # 2. Route
        decision = self.router.route(query, chunks)
        if verbose:
            print(f"\n[Agent] Routing decision: {decision.explain()}")

        # 3. Generate
        result   = self.generator.generate(query, decision, verbose=verbose)
        result["query"] = query
        return result

    # ------------------------------------------------------------------ #
    def ask_pretty(self, query: str) -> None:
        result = self.ask(query, verbose=True)
        print("\n" + "=" * 70)
        print(f"Query      : {result['query']}")
        print(f"Type       : {result['query_type']}")
        print(f"Routing    : {result['routing_reason']}")
        print(f"Sources    : {', '.join(result['sources']) or 'none'}")
        print(f"Chunks used: {result['chunks_used']}")
        print("-" * 70)
        print(result['answer'])
        print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    agent = QAAgent()
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        data_dir  = sys.argv[2] if len(sys.argv) > 2 else "data"
        index_dir = sys.argv[3] if len(sys.argv) > 3 else "index"
        agent.ingest(data_dir, index_dir)
    else:
        query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the EU AI Act?"
        agent.ask_pretty(query)
