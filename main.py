#!/usr/bin/env python3
"""
main.py — CLI entry point for the Agentic Q&A System.

Usage:
  python main.py ingest [data_dir] [index_dir]
  python main.py ask "your question here"
  python main.py eval [--verbose]
  python main.py failures
  python main.py demo
"""

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))


def cmd_ingest(args):
    from src.agent import QAAgent
    data_dir  = args[0] if args else "data"
    index_dir = args[1] if len(args) > 1 else "index"
    agent = QAAgent(index_dir=index_dir)
    agent.ingest(data_dir, index_dir)
    print(f"\nIngestion complete. Index saved to '{index_dir}/'")


def cmd_ask(args):
    from src.agent import QAAgent
    if not args:
        print("Usage: python main.py ask \"<your question>\"")
        sys.exit(1)
    query = " ".join(args)
    agent = QAAgent()
    agent.ask_pretty(query)


def cmd_eval(args):
    from src.agent   import QAAgent
    from eval.evaluate import run_evaluation, print_results_table
    verbose = "--verbose" in args or "-v" in args
    agent = QAAgent()
    rows  = run_evaluation(agent, verbose=verbose)
    print_results_table(rows)


def cmd_failures(_args):
    from eval.failure_analysis import print_failure_report
    print_failure_report()


def cmd_demo(_args):
    """Run a handful of representative questions interactively."""
    from src.agent import QAAgent
    agent = QAAgent()
    demo_queries = [
        "What is the EU AI Act?",
        "Compare how the EU and US approach AI regulation.",
        "What is the best recipe for pasta carbonara?",
    ]
    for q in demo_queries:
        agent.ask_pretty(q)
        print()


COMMANDS = {
    "ingest":   cmd_ingest,
    "ask":      cmd_ask,
    "eval":     cmd_eval,
    "failures": cmd_failures,
    "demo":     cmd_demo,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Available commands:", ", ".join(COMMANDS))
        sys.exit(1)
    cmd  = sys.argv[1]
    rest = sys.argv[2:]
    COMMANDS[cmd](rest)
