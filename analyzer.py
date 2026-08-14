"""Core ticket-analysis logic: builds prompts, calls the LLM, and parses
its JSON response into a validated AnalysisResult."""
import json
import re
from typing import Any, Dict

from models import AnalysisResult, Ticket
from ollama_client import OllamaClient

SYSTEM_PROMPT = (
    "You are an expert customer support ticket analyst. You always respond "
    "with strict, valid JSON and nothing else - no markdown fences, no "
    "commentary before or after the JSON object."
)

CATEGORIES = [
    "Billing", "Technical Issue", "Account Access", "Feature Request",
    "Bug Report", "Shipping/Delivery", "Refund/Return", "General Inquiry",
    "Complaint", "Other",
]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
SENTIMENTS = ["Positive", "Neutral", "Negative", "Very Negative"]


def build_prompt(ticket: Ticket) -> str:
    return f"""Analyze the following customer support ticket and respond with ONLY a JSON object
matching this exact schema:

{{
  "category": one of {CATEGORIES},
  "sentiment": one of {SENTIMENTS},
  "priority": one of {PRIORITIES},
  "summary": "one to two sentence summary of the issue",
  "suggested_response": "a short, empathetic draft reply to the customer (3-5 sentences)",
  "keywords": ["up to 5 short keywords/phrases relevant to the ticket"]
}}

Ticket ID: {ticket.ticket_id}
Subject: {ticket.subject}
Description:
{ticket.description}

Respond with ONLY the JSON object, no other text.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """Models occasionally wrap JSON in markdown fences or add stray text
    around it. Strip fences, then grab the first {...} block and parse it."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))


class TicketAnalyzer:
    def __init__(self, client: OllamaClient):
        self.client = client

    def analyze(self, ticket: Ticket) -> AnalysisResult:
        prompt = build_prompt(ticket)
        raw = self.client.generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)

        try:
            data = _extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            # Degrade gracefully rather than crashing the whole batch run.
            return AnalysisResult(
                ticket_id=ticket.ticket_id,
                category="Other",
                sentiment="Neutral",
                priority="Medium",
                summary="Could not be parsed automatically - needs manual review.",
                suggested_response="",
                keywords=[],
                confidence_notes="PARSE_ERROR",
                raw_model_output=raw,
            )

        category = data.get("category", "Other")
        sentiment = data.get("sentiment", "Neutral")
        priority = data.get("priority", "Medium")

        # Guard against the model inventing values outside our schema.
        if category not in CATEGORIES:
            category = "Other"
        if sentiment not in SENTIMENTS:
            sentiment = "Neutral"
        if priority not in PRIORITIES:
            priority = "Medium"

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        return AnalysisResult(
            ticket_id=ticket.ticket_id,
            category=category,
            sentiment=sentiment,
            priority=priority,
            summary=str(data.get("summary", "")).strip(),
            suggested_response=str(data.get("suggested_response", "")).strip(),
            keywords=[str(k) for k in keywords][:5],
            raw_model_output=raw,
        )