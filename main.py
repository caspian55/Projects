#!/usr/bin/env python3
"""
Customer Support Ticket Analyzer
---------------------------------
Uses a local Ollama LLM to classify, score, and summarize customer support
tickets in bulk, then produces CSV/JSON output plus a summary report.

Usage:
    python main.py --input sample_data/sample_tickets.csv --output results

Requirements:
    - Ollama installed and running locally (https://ollama.com)
    - A pulled model, e.g.: ollama pull llama3.1
    - pip install -r requirements.txt
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

from analyzer import TicketAnalyzer
from models import Ticket
from ollama_client import OllamaClient
from report_generator import build_summary


def read_tickets(path: Path) -> List[Ticket]:
    tickets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ticket_id", "subject", "description"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"Input CSV is missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            tickets.append(
                Ticket(
                    ticket_id=row["ticket_id"].strip(),
                    subject=row["subject"].strip(),
                    description=row["description"].strip(),
                    customer_email=(row.get("customer_email") or "").strip() or None,
                    created_at=(row.get("created_at") or "").strip() or None,
                )
            )
    return tickets


def write_csv(results, path: Path) -> None:
    fieldnames = [
        "ticket_id", "category", "sentiment", "priority",
        "summary", "suggested_response", "keywords", "confidence_notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["keywords"] = "; ".join(row["keywords"])
            row.pop("raw_model_output", None)
            writer.writerow(row)


def write_json(results, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze customer support tickets with a local Ollama model."
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Path to input CSV (ticket_id, subject, description[, customer_email, created_at])")
    parser.add_argument("--output", "-o", default="results",
                         help="Output file prefix (default: results)")
    parser.add_argument("--model", "-m", default="llama3.1",
                         help="Ollama model name (default: llama3.1)")
    parser.add_argument("--host", default="http://localhost:11434",
                         help="Ollama host URL (default: http://localhost:11434)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N tickets (useful for testing)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    client = OllamaClient(model=args.model, host=args.host)
    if not client.is_available():
        sys.exit(
            f"Could not reach Ollama at {args.host}.\n"
            f"  1. Make sure Ollama is installed: https://ollama.com\n"
            f"  2. Start the server: ollama serve\n"
            f"  3. Pull the model:   ollama pull {args.model}"
        )

    tickets = read_tickets(input_path)
    if args.limit:
        tickets = tickets[: args.limit]

    if not tickets:
        sys.exit("No tickets found in input file.")

    print(f"Loaded {len(tickets)} ticket(s) from {input_path}")
    print(f"Using model '{args.model}' at {args.host}\n")

    analyzer = TicketAnalyzer(client)
    results = []
    start = time.time()

    for i, ticket in enumerate(tickets, 1):
        preview = (ticket.subject[:50] + "...") if len(ticket.subject) > 50 else ticket.subject
        print(f"[{i}/{len(tickets)}] Analyzing {ticket.ticket_id}: {preview}")
        try:
            results.append(analyzer.analyze(ticket))
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            sys.exit(f"\nStopped: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(tickets):.1f}s/ticket avg)\n")

    csv_path = Path(f"{args.output}.csv")
    json_path = Path(f"{args.output}.json")
    report_path = Path(f"{args.output}_report.txt")

    write_csv(results, csv_path)
    write_json(results, json_path)

    summary = build_summary(results)
    report_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nSaved: {csv_path}, {json_path}, {report_path}")


if __name__ == "__main__":
    main()