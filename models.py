"""Data models used throughout the ticket analyzer."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Ticket:
    """A single customer support ticket loaded from the input CSV."""
    ticket_id: str
    subject: str
    description: str
    customer_email: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class AnalysisResult:
    """The structured output produced by the LLM for one ticket."""
    ticket_id: str
    category: str
    sentiment: str
    priority: str
    summary: str
    suggested_response: str
    keywords: List[str] = field(default_factory=list)
    confidence_notes: str = ""
    raw_model_output: str = ""