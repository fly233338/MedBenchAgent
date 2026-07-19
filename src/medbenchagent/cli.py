"""Command-line entry points for workflow development and benchmark finalization."""

from __future__ import annotations

import argparse

from medbenchagent.answer_balance import balance_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="medbenchagent")
    subcommands = parser.add_subparsers(dest="command")
    balance = subcommands.add_parser("balance-answers", help="Write a balanced JSONL copy.")
    balance.add_argument("--input", required=True, dest="input_path")
    balance.add_argument("--output", required=True, dest="output_path")
    balance.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "balance-answers":
        print(balance_jsonl(args.input_path, args.output_path, args.seed))
        return
    print("Start the interactive planner with: uv run langgraph dev")
