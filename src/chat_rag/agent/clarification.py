"""
Clarifying Questions Module

Detects ambiguous queries and generates clarifying questions
before executing retrieval. Improves answer quality by gathering
missing context upfront.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class AmbiguityType(Enum):
    """Types of ambiguity that require clarification."""

    NONE = "none"
    MISSING_COUNTRY = "missing_country"
    MISSING_CONTEXT = "missing_context"
    MULTIPLE_TOPICS = "multiple_topics"
    VAGUE_REFERENCE = "vague_reference"
    TIME_AMBIGUOUS = "time_ambiguous"


@dataclass
class ClarificationResult:
    """Result of ambiguity detection."""

    needs_clarification: bool
    ambiguity_type: AmbiguityType
    clarifying_question: Optional[str] = None
    detected_topics: List[str] = None
    confidence: float = 1.0


AMBIGUITY_DETECTION_PROMPT = """Analyze this HR query for ambiguity. Determine if clarification is needed before answering.

Query: "{query}"
User's country: {country}
Conversation context: {context}

Check for these ambiguity types:
1. MISSING_COUNTRY - Query is country-specific but country unknown (e.g., "What's the parental leave?")
2. MISSING_CONTEXT - Needs personal info not available (e.g., "Am I eligible?" without knowing role/tenure)
3. MULTIPLE_TOPICS - Query spans multiple distinct policy areas
4. VAGUE_REFERENCE - Uses pronouns/references without antecedent ("that policy", "the form")
5. TIME_AMBIGUOUS - Unclear time reference ("recently", "upcoming")
6. NONE - Query is clear enough to answer

Respond in this exact format:
AMBIGUITY_TYPE: <type>
NEEDS_CLARIFICATION: <true/false>
CLARIFYING_QUESTION: <question to ask, or "none">
DETECTED_TOPICS: <comma-separated topics>
CONFIDENCE: <0.0-1.0>"""


class ClarificationDetector:
    """
    Detects when queries need clarification before retrieval.

    Usage:
        detector = ClarificationDetector()
        result = await detector.analyze(query, user_country, context)
        if result.needs_clarification:
            return result.clarifying_question
    """

    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Quick heuristic patterns (avoid LLM call for obvious cases)
        self.country_specific_keywords = {
            "parental leave",
            "maternity",
            "paternity",
            "sick leave",
            "public holiday",
            "tax",
            "pension",
            "healthcare",
            "visa",
        }
        self.vague_patterns = {"that policy", "the form", "this process", "that thing", "it", "they", "those"}

    def _quick_check(self, query: str, user_country: Optional[str]) -> Optional[ClarificationResult]:
        """Fast heuristic check before LLM call."""
        query_lower = query.lower()

        # Check for country-specific topics without known country
        if not user_country or user_country.lower() == "unknown":
            for keyword in self.country_specific_keywords:
                if keyword in query_lower:
                    return ClarificationResult(
                        needs_clarification=True,
                        ambiguity_type=AmbiguityType.MISSING_COUNTRY,
                        clarifying_question=f"Policies for {keyword} vary by country. Which country are you asking about?",
                        confidence=0.9,
                    )

        # Check for vague references at start of conversation
        for pattern in self.vague_patterns:
            if pattern in query_lower and query_lower.index(pattern) < 20:
                return ClarificationResult(
                    needs_clarification=True,
                    ambiguity_type=AmbiguityType.VAGUE_REFERENCE,
                    clarifying_question="I'm not sure what you're referring to. Could you be more specific?",
                    confidence=0.85,
                )

        return None

    async def analyze(
        self,
        query: str,
        user_country: Optional[str] = None,
        conversation_context: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
    ) -> ClarificationResult:
        """
        Analyze query for ambiguity.

        Args:
            query: User's question
            user_country: Known country (from profile)
            conversation_context: Recent conversation history
            system_prompt_override: Optional override for the detection prompt

        Returns:
            ClarificationResult with analysis
        """
        # Try quick heuristic first
        quick_result = self._quick_check(query, user_country)
        if quick_result:
            logger.info(f"Quick check detected: {quick_result.ambiguity_type.value}")
            return quick_result

        # Fall back to LLM analysis for complex cases
        context_str = conversation_context or "No prior context"
        country_str = user_country or "Unknown"

        logger.info(f"Analyzing ambiguity for query='{query}' with country='{country_str}'")

        prompt_template = system_prompt_override or AMBIGUITY_DETECTION_PROMPT
        prompt = prompt_template.format(
            query=query,
            country=country_str,
            context=context_str[:500],
        )

        logger.debug(f"Ambiguity detection prompt: {prompt}")

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You analyze HR queries for ambiguity. Use the provided context (country, etc.) to avoid unnecessary clarification."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            logger.info(f"Raw ambiguity detection response: {response.content}")
            return self._parse_response(response.content)

        except Exception as e:
            logger.error(f"Clarification detection failed: {e}")
            return ClarificationResult(
                needs_clarification=False,
                ambiguity_type=AmbiguityType.NONE,
            )

    def _parse_response(self, response: str) -> ClarificationResult:
        """Parse LLM response into ClarificationResult."""
        lines = response.strip().split("\n")
        parsed = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.strip().upper()] = value.strip()

        ambiguity_type = AmbiguityType.NONE
        try:
            ambiguity_type = AmbiguityType(parsed.get("AMBIGUITY_TYPE", "none").lower())
        except ValueError:
            pass

        needs_clarification = parsed.get("NEEDS_CLARIFICATION", "false").lower() == "true"
        clarifying_question = parsed.get("CLARIFYING_QUESTION")
        if clarifying_question and clarifying_question.lower() == "none":
            clarifying_question = None

        topics = []
        if "DETECTED_TOPICS" in parsed:
            topics = [t.strip() for t in parsed["DETECTED_TOPICS"].split(",")]

        confidence = 0.8
        try:
            confidence = float(parsed.get("CONFIDENCE", "0.8"))
        except ValueError:
            pass

        return ClarificationResult(
            needs_clarification=needs_clarification,
            ambiguity_type=ambiguity_type,
            clarifying_question=clarifying_question,
            detected_topics=topics,
            confidence=confidence,
        )


# Pre-built clarifying questions for common ambiguities
CLARIFYING_TEMPLATES = {
    AmbiguityType.MISSING_COUNTRY: [
        "Policies vary by country. Are you asking about Switzerland (CH) or Italy (IT)?",
        "Which country's policy would you like me to look up?",
    ],
    AmbiguityType.MISSING_CONTEXT: [
        "To give you accurate information, could you tell me your role or employment type?",
        "This depends on your tenure. How long have you been with the company?",
    ],
    AmbiguityType.MULTIPLE_TOPICS: [
        "Your question touches on multiple topics. Would you like me to address {topics} separately?",
        "I found information on {topics}. Which would you like to explore first?",
    ],
    AmbiguityType.TIME_AMBIGUOUS: [
        "Are you asking about current policies or upcoming changes?",
        "Could you clarify the time period you're asking about?",
    ],
}
