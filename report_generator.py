"""Builds a plain-text summary report from a batch of AnalysisResults."""
from collections import Counter
from typing import List

from models import AnalysisResult


def build_summary(results: List[AnalysisResult]) -> str:
    total = len(results)
    if total == 0:
        return "No tickets analyzed."

    cat_counts = Counter(r.category for r in results)
    sentiment_counts = Counter(r.sentiment for r in results)
    priority_counts = Counter(r.priority for r in results)
    urgent = [r for r in results if r.priority in ("High", "Urgent")]
    parse_errors = sum(1 for r in results if r.confidence_notes == "PARSE_ERROR")

    lines = []
    lines.append("=" * 60)
    lines.append("CUSTOMER SUPPORT TICKET ANALYSIS - SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append(f"Total tickets analyzed: {total}")
    if parse_errors:
        lines.append(f"Tickets needing manual review (parse errors): {parse_errors}")
    lines.append("")

    lines.append("By category:")
    for cat, count in cat_counts.most_common():
        lines.append(f"  {cat:<20} {count:>4}  ({count / total * 100:5.1f}%)")

    lines.append("\nBy sentiment:")
    for s, count in sentiment_counts.most_common():
        lines.append(f"  {s:<20} {count:>4}  ({count / total * 100:5.1f}%)")

    lines.append("\nBy priority:")
    for p in ["Urgent", "High", "Medium", "Low"]:
        count = priority_counts.get(p, 0)
        lines.append(f"  {p:<20} {count:>4}  ({count / total * 100:5.1f}%)")

    if urgent:
        lines.append(f"\n{len(urgent)} ticket(s) flagged High/Urgent priority:")
        for r in urgent:
            lines.append(f"  - [{r.ticket_id}] {r.summary}")

    lines.append("=" * 60)
    return "\n".join(lines)