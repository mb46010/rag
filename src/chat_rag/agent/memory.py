"""
Multi-turn Session Memory

Maintains conversation context within a session to handle:
- Follow-up questions ("what about for Italy?")
- Pronoun resolution ("how do I apply for it?")
- Topic continuity ("and the deadline?")

This is session-scoped memory, not persistent across sessions.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Extracted context
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # e.g., {"country": "CH"}
    query_type: Optional[str] = None
    
    # Retrieved info (for assistant turns)
    sources_used: List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Accumulated context from the session."""
    # Current focus
    active_topic: Optional[str] = None
    active_country: Optional[str] = None
    active_policy: Optional[str] = None
    
    # Entities mentioned
    mentioned_policies: List[str] = field(default_factory=list)
    mentioned_countries: List[str] = field(default_factory=list)
    
    # For reference resolution
    last_policy_discussed: Optional[str] = None
    last_country_discussed: Optional[str] = None
    pending_clarification: Optional[str] = None


class SessionMemory:
    """
    Manages conversation context within a session.
    
    Features:
    - Tracks conversation history (last N turns)
    - Extracts and maintains entities (countries, policies, etc.)
    - Resolves references ("it", "that policy", "there")
    - Rewrites queries with resolved context
    
    Usage:
        memory = SessionMemory()
        memory.add_user_message("What's the PTO policy in Switzerland?")
        memory.add_assistant_message("The PTO policy...", topics=["PTO"], entities={"country": "CH"})
        
        # Later...
        resolved = memory.resolve_query("What about in Italy?")
        # Returns: "What is the PTO policy in Italy?"
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: Maximum conversation turns to keep
        """
        self.max_turns = max_turns
        self.history: deque[ConversationTurn] = deque(maxlen=max_turns)
        self.context = SessionContext()
    
    def add_user_message(
        self,
        content: str,
        topics: List[str] = None,
        entities: Dict[str, str] = None,
    ):
        """Add a user message to history."""
        turn = ConversationTurn(
            role="user",
            content=content,
            topics=topics or [],
            entities=entities or {},
        )
        self.history.append(turn)
        
        # Update context
        if entities:
            self._update_context_from_entities(entities)
        if topics:
            self.context.active_topic = topics[0] if topics else None
            self.context.mentioned_policies.extend(topics)
    
    def add_assistant_message(
        self,
        content: str,
        topics: List[str] = None,
        entities: Dict[str, str] = None,
        sources: List[str] = None,
    ):
        """Add an assistant message to history."""
        turn = ConversationTurn(
            role="assistant",
            content=content,
            topics=topics or [],
            entities=entities or {},
            sources_used=sources or [],
        )
        self.history.append(turn)
        
        # Update context
        if entities:
            self._update_context_from_entities(entities)
        if topics:
            self.context.last_policy_discussed = topics[0] if topics else None
        if sources:
            self.context.mentioned_policies.extend(sources)
    
    def _update_context_from_entities(self, entities: Dict[str, str]):
        """Update session context from extracted entities."""
        if "country" in entities:
            self.context.active_country = entities["country"]
            self.context.last_country_discussed = entities["country"]
            if entities["country"] not in self.context.mentioned_countries:
                self.context.mentioned_countries.append(entities["country"])
        
        if "policy" in entities:
            self.context.active_policy = entities["policy"]
            self.context.last_policy_discussed = entities["policy"]
    
    def resolve_query(self, query: str) -> str:
        """
        Resolve references in query using conversation context.
        
        Examples:
            "what about Italy?" → "What is the PTO policy in Italy?"
            "how do I apply for it?" → "How do I apply for parental leave?"
            "and the deadline?" → "What is the deadline for expense reports?"
        """
        resolved = query
        query_lower = query.lower()
        
        # Detect follow-up patterns
        is_followup = any(pattern in query_lower for pattern in [
            "what about", "how about", "and ", "also ", "what if",
            "same for", "in that case", "for that"
        ])
        
        # Resolve country references
        country_refs = ["there", "that country", "the same country"]
        for ref in country_refs:
            if ref in query_lower and self.context.last_country_discussed:
                resolved = resolved.replace(ref, self.context.last_country_discussed)
        
        # Resolve "what about [country]" pattern
        if "what about" in query_lower:
            # Check if it's a country switch
            countries = {"italy": "IT", "switzerland": "CH", "ch": "CH", "it": "IT"}
            for country_name, code in countries.items():
                if country_name in query_lower:
                    if self.context.active_topic:
                        resolved = f"What is the {self.context.active_topic} policy in {code}?"
                    break
        
        # Resolve policy references
        policy_refs = ["it", "that", "this policy", "that policy", "the policy"]
        for ref in policy_refs:
            # Only replace if it's likely referring to a policy
            if f" {ref}" in query_lower or query_lower.startswith(ref):
                if self.context.last_policy_discussed:
                    # Be careful with "it" - only replace in certain patterns
                    if ref == "it" and "apply for it" in query_lower:
                        resolved = resolved.replace("for it", f"for {self.context.last_policy_discussed}")
                    elif ref != "it":
                        resolved = resolved.replace(ref, self.context.last_policy_discussed)
        
        # Handle incomplete follow-ups like "and the deadline?"
        if query_lower.startswith("and ") and self.context.active_topic:
            # Extract what they're asking about
            remainder = query[4:].strip()
            resolved = f"What is {remainder} for {self.context.active_topic}?"
        
        if resolved != query:
            logger.info(f"Resolved query: '{query}' → '{resolved}'")
        
        return resolved
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for the LLM."""
        parts = []
        
        if self.context.active_topic:
            parts.append(f"Current topic: {self.context.active_topic}")
        
        if self.context.active_country:
            parts.append(f"Country context: {self.context.active_country}")
        
        if self.context.last_policy_discussed:
            parts.append(f"Last policy discussed: {self.context.last_policy_discussed}")
        
        # Recent conversation summary
        if len(self.history) > 0:
            recent = list(self.history)[-3:]  # Last 3 turns
            convo = " | ".join([f"{t.role}: {t.content[:50]}..." for t in recent])
            parts.append(f"Recent conversation: {convo}")
        
        return " | ".join(parts) if parts else "No prior context"
    
    def get_history_for_llm(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM context."""
        recent = list(self.history)[-max_turns:]
        return [
            {"role": turn.role, "content": turn.content}
            for turn in recent
        ]
    
    def extract_entities_from_query(self, query: str) -> Dict[str, str]:
        """Extract entities from a query (simple pattern matching)."""
        entities = {}
        query_lower = query.lower()
        
        # Country extraction
        country_patterns = {
            "switzerland": "CH", "swiss": "CH", "ch": "CH",
            "italy": "IT", "italian": "IT", "it": "IT",
        }
        for pattern, code in country_patterns.items():
            if pattern in query_lower:
                entities["country"] = code
                break
        
        # Policy/topic extraction
        topic_patterns = {
            "pto": "PTO", "vacation": "PTO", "holiday": "PTO", "time off": "PTO",
            "expense": "Expenses", "reimbursement": "Expenses",
            "wfh": "WFH", "work from home": "WFH", "remote": "WFH",
            "parental": "Parental Leave", "maternity": "Parental Leave", "paternity": "Parental Leave",
            "sick": "Sick Leave", "illness": "Sick Leave",
        }
        for pattern, topic in topic_patterns.items():
            if pattern in query_lower:
                entities["topic"] = topic
                break
        
        return entities
    
    def clear(self):
        """Clear all memory."""
        self.history.clear()
        self.context = SessionContext()
    
    @property
    def turn_count(self) -> int:
        """Number of turns in history."""
        return len(self.history)
    
    @property
    def has_context(self) -> bool:
        """Check if there's meaningful context."""
        return (
            self.context.active_topic is not None or
            self.context.active_country is not None or
            len(self.history) > 0
        )
