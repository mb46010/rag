"""
Source Highlighting Module

Identifies and highlights the exact text spans within retrieved chunks
that support the generated answer. Provides precise citations rather
than just showing entire chunks.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class HighlightedSpan:
    """A highlighted span within a source document."""
    text: str
    start_char: int
    end_char: int
    relevance_score: float = 1.0
    
    def __str__(self):
        return f'"{self.text}"'


@dataclass
class HighlightedSource:
    """A source document with highlighted relevant spans."""
    document_name: str
    section_path: str
    full_text: str
    highlighted_spans: List[HighlightedSpan] = field(default_factory=list)
    chunk_id: str = ""
    
    @property
    def highlighted_text(self) -> str:
        """Return text with spans marked using **bold**."""
        if not self.highlighted_spans:
            return self.full_text
        
        # Sort spans by start position (reverse for safe insertion)
        sorted_spans = sorted(self.highlighted_spans, key=lambda s: s.start_char, reverse=True)
        
        result = self.full_text
        for span in sorted_spans:
            before = result[:span.start_char]
            highlighted = f"**{result[span.start_char:span.end_char]}**"
            after = result[span.end_char:]
            result = before + highlighted + after
        
        return result
    
    @property
    def excerpt(self) -> str:
        """Return just the highlighted spans as excerpts."""
        if not self.highlighted_spans:
            return self.full_text[:200] + "..."
        
        excerpts = [f'"{span.text}"' for span in self.highlighted_spans[:3]]
        return " ... ".join(excerpts)


EXTRACTION_PROMPT = """Given the user's question and an AI-generated answer, identify the exact text spans from the source that support the answer.

Question: {question}

Answer: {answer}

Source text:
{source_text}

Extract 1-3 specific text spans (exact quotes) from the source that directly support the answer.
Return each span on a new line, exactly as it appears in the source.
Only include spans that are DIRECTLY relevant to answering the question.

Relevant spans:"""


class SourceHighlighter:
    """
    Identifies exact text spans that support an answer.
    
    Usage:
        highlighter = SourceHighlighter()
        sources = highlighter.highlight(question, answer, chunks)
        for source in sources:
            print(source.highlighted_text)
    """
    
    def __init__(self, llm: ChatOpenAI = None, use_llm: bool = True):
        """
        Args:
            llm: LLM for span extraction (optional)
            use_llm: If False, use heuristic matching only
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.use_llm = use_llm
    
    async def highlight(
        self,
        question: str,
        answer: str,
        chunks: List,  # List[PolicyChunk]
    ) -> List[HighlightedSource]:
        """
        Highlight relevant spans in retrieved chunks.
        
        Args:
            question: User's question
            answer: Generated answer
            chunks: Retrieved PolicyChunks
            
        Returns:
            List of HighlightedSource with marked spans
        """
        results = []
        
        for chunk in chunks:
            if self.use_llm:
                spans = await self._extract_spans_llm(question, answer, chunk.text)
            else:
                spans = self._extract_spans_heuristic(question, answer, chunk.text)
            
            highlighted = HighlightedSource(
                document_name=chunk.document_name,
                section_path=chunk.section_path_str,
                full_text=chunk.text,
                highlighted_spans=spans,
                chunk_id=chunk.chunk_id,
            )
            results.append(highlighted)
        
        return results
    
    def highlight_sync(
        self,
        question: str,
        answer: str,
        chunks: List,
    ) -> List[HighlightedSource]:
        """Synchronous version using heuristics only."""
        results = []
        
        for chunk in chunks:
            spans = self._extract_spans_heuristic(question, answer, chunk.text)
            
            highlighted = HighlightedSource(
                document_name=chunk.document_name,
                section_path=chunk.section_path_str,
                full_text=chunk.text,
                highlighted_spans=spans,
                chunk_id=chunk.chunk_id,
            )
            results.append(highlighted)
        
        return results
    
    async def _extract_spans_llm(
        self,
        question: str,
        answer: str,
        source_text: str,
    ) -> List[HighlightedSpan]:
        """Use LLM to extract relevant spans."""
        prompt = EXTRACTION_PROMPT.format(
            question=question,
            answer=answer,
            source_text=source_text,
        )
        
        try:
            response = await self.llm.ainvoke([
                HumanMessage(content=prompt)
            ])
            
            spans = []
            for line in response.content.strip().split("\n"):
                line = line.strip().strip('"').strip("'")
                if line and len(line) > 10:
                    span = self._find_span_in_text(line, source_text)
                    if span:
                        spans.append(span)
            
            return spans[:3]  # Max 3 spans per chunk
            
        except Exception as e:
            logger.warning(f"LLM span extraction failed: {e}")
            return self._extract_spans_heuristic(question, answer, source_text)
    
    def _extract_spans_heuristic(
        self,
        question: str,
        answer: str,
        source_text: str,
    ) -> List[HighlightedSpan]:
        """
        Heuristic span extraction based on:
        1. Sentences containing question keywords
        2. Sentences with numbers/dates (often key facts)
        3. Sentences that appear (paraphrased) in answer
        """
        spans = []
        sentences = self._split_sentences(source_text)
        
        # Extract keywords from question
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Check question keyword overlap
            for keyword in question_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Check answer keyword overlap
            for keyword in answer_keywords:
                if keyword in sentence_lower:
                    score += 1
            
            # Bonus for numbers/dates (often key policy details)
            if re.search(r'\d+', sentence):
                score += 1
            
            if score >= 2:
                span = self._find_span_in_text(sentence, source_text)
                if span:
                    span.relevance_score = min(score / 5, 1.0)
                    spans.append(span)
        
        # Sort by relevance and take top 3
        spans.sort(key=lambda s: s.relevance_score, reverse=True)
        return spans[:3]
    
    def _find_span_in_text(self, span_text: str, full_text: str) -> Optional[HighlightedSpan]:
        """Find exact position of span in source text."""
        # Try exact match first
        start = full_text.find(span_text)
        if start != -1:
            return HighlightedSpan(
                text=span_text,
                start_char=start,
                end_char=start + len(span_text),
            )
        
        # Try case-insensitive match
        start = full_text.lower().find(span_text.lower())
        if start != -1:
            actual_text = full_text[start:start + len(span_text)]
            return HighlightedSpan(
                text=actual_text,
                start_char=start,
                end_char=start + len(span_text),
            )
        
        # Try fuzzy match (first 50 chars)
        if len(span_text) > 50:
            truncated = span_text[:50]
            start = full_text.lower().find(truncated.lower())
            if start != -1:
                # Extend to sentence end
                end = full_text.find(".", start + 50)
                if end == -1:
                    end = min(start + len(span_text), len(full_text))
                actual_text = full_text[start:end + 1]
                return HighlightedSpan(
                    text=actual_text,
                    start_char=start,
                    end_char=end + 1,
                )
        
        return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "i", "you",
            "we", "they", "it", "my", "your", "our", "their", "this", "that",
            "what", "how", "many", "much", "get", "for", "to", "of", "in",
            "on", "at", "with", "and", "or", "but"
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords


def format_highlighted_sources(sources: List[HighlightedSource]) -> str:
    """Format highlighted sources for display in chat."""
    if not sources:
        return ""
    
    output = "\n\n📚 **Sources:**\n"
    
    for i, source in enumerate(sources, 1):
        output += f"\n**[{i}] {source.document_name}** - {source.section_path}\n"
        
        if source.highlighted_spans:
            output += "Key excerpts:\n"
            for span in source.highlighted_spans:
                output += f"  → _{span.text[:150]}{'...' if len(span.text) > 150 else ''}_\n"
        else:
            output += f"  {source.full_text[:150]}...\n"
    
    return output
