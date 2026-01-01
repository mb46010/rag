"""
Synthetic Evaluation Set Generator

Generates Q&A pairs from policy documents for bootstrapping evaluation.
Part of the Hybrid approach (Option 4).

Usage:
    generator = SyntheticQAGenerator()
    qa_pairs = await generator.generate_from_documents("documents/")
    generator.export_for_review("eval_draft.json")
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions to generate."""
    FACTUAL = "factual"           # "How many days...?"
    PROCEDURAL = "procedural"     # "How do I...?"
    ELIGIBILITY = "eligibility"   # "Am I eligible for...?"
    COMPARISON = "comparison"     # "What's the difference between...?"
    CONDITIONAL = "conditional"   # "What if...?"
    NEGATION = "negation"         # "What is NOT covered...?"


@dataclass
class GeneratedQA:
    """A generated Q&A pair for evaluation."""
    question: str
    expected_answer: str
    source_document: str
    source_section: str
    question_type: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    
    # For expert review
    expert_validated: bool = False
    expert_notes: str = ""
    quality_score: Optional[int] = None  # 1-5
    
    # Metadata
    country_specific: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)


GENERATION_PROMPT = """You are generating evaluation questions for an HR policy RAG system.

Given this policy document section, generate {n_questions} diverse questions that an employee might ask.

Document: {doc_name}
Section: {section_name}
Content:
{content}

Generate questions of these types (aim for variety):
- FACTUAL: Specific facts, numbers, limits, dates
- PROCEDURAL: How to do something, steps, process
- ELIGIBILITY: Who qualifies, requirements
- CONDITIONAL: What-if scenarios
- NEGATION: What's NOT allowed/covered

For each question, provide:
1. The question an employee would naturally ask
2. The expected answer (based on the content)
3. The question type
4. Difficulty (easy/medium/hard)

Format as JSON array:
[
  {{
    "question": "...",
    "expected_answer": "...",
    "question_type": "factual|procedural|eligibility|conditional|negation",
    "difficulty": "easy|medium|hard"
  }}
]

Generate realistic, natural questions - not robotic or overly formal."""


class SyntheticQAGenerator:
    """
    Generates synthetic Q&A pairs from policy documents.
    
    Workflow:
    1. Load policy documents
    2. Generate Q&A pairs per section
    3. Export for expert review
    4. Import validated pairs into eval set
    """
    
    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.generated_pairs: List[GeneratedQA] = []
    
    async def generate_from_documents(
        self,
        docs_dir: str,
        questions_per_section: int = 3,
    ) -> List[GeneratedQA]:
        """
        Generate Q&A pairs from all documents in directory.
        
        Args:
            docs_dir: Path to policy documents (JSON format)
            questions_per_section: Number of questions per section
            
        Returns:
            List of GeneratedQA pairs
        """
        docs_path = Path(docs_dir)
        all_pairs = []
        
        for doc_file in docs_path.glob("*.json"):
            logger.info(f"Processing: {doc_file.name}")
            
            with open(doc_file) as f:
                doc = json.load(f)
            
            doc_name = doc.get("metadata", {}).get("document_name", doc_file.stem)
            country = doc.get("metadata", {}).get("country")
            content = doc.get("content", "")
            
            # Split into sections (simple markdown header split)
            sections = self._split_into_sections(content)
            
            for section_name, section_content in sections:
                if len(section_content.strip()) < 100:
                    continue  # Skip tiny sections
                
                pairs = await self._generate_for_section(
                    doc_name=doc_name,
                    section_name=section_name,
                    content=section_content,
                    country=country,
                    n_questions=questions_per_section,
                )
                all_pairs.extend(pairs)
        
        self.generated_pairs = all_pairs
        logger.info(f"Generated {len(all_pairs)} Q&A pairs")
        return all_pairs
    
    async def _generate_for_section(
        self,
        doc_name: str,
        section_name: str,
        content: str,
        country: Optional[str],
        n_questions: int,
    ) -> List[GeneratedQA]:
        """Generate Q&A pairs for a single section."""
        prompt = GENERATION_PROMPT.format(
            doc_name=doc_name,
            section_name=section_name,
            content=content[:3000],  # Limit context
            n_questions=n_questions,
        )
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="Generate evaluation Q&A pairs in JSON format."),
                HumanMessage(content=prompt),
            ])
            
            # Parse JSON from response
            pairs_data = self._parse_json_response(response.content)
            
            pairs = []
            for item in pairs_data:
                qa = GeneratedQA(
                    question=item["question"],
                    expected_answer=item["expected_answer"],
                    source_document=doc_name,
                    source_section=section_name,
                    question_type=item.get("question_type", "factual"),
                    difficulty=item.get("difficulty", "medium"),
                    country_specific=country,
                    tags=[doc_name.lower().replace(" ", "_")],
                )
                pairs.append(qa)
            
            return pairs
            
        except Exception as e:
            logger.error(f"Generation failed for {doc_name}/{section_name}: {e}")
            return []
    
    def _split_into_sections(self, content: str) -> List[tuple]:
        """Split markdown content into (section_name, content) tuples."""
        import re
        
        sections = []
        current_section = "Overview"
        current_content = []
        
        for line in content.split("\n"):
            header_match = re.match(r'^#{1,3}\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append((current_section, "\n".join(current_content)))
                current_section = header_match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections.append((current_section, "\n".join(current_content)))
        
        return sections
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Extract JSON array from LLM response."""
        import re
        
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            return json.loads(json_match.group())
        
        # Fallback: try parsing entire response
        return json.loads(response)
    
    def export_for_review(self, filepath: str):
        """
        Export generated pairs for expert review.
        
        Creates a JSON file with all pairs marked as unvalidated.
        Experts can edit this file to:
        - Fix incorrect expected answers
        - Adjust difficulty ratings
        - Add notes
        - Mark quality scores
        - Delete bad questions
        """
        export_data = {
            "metadata": {
                "generated_count": len(self.generated_pairs),
                "review_instructions": """
                    For each Q&A pair:
                    1. Check if question is realistic (would an employee ask this?)
                    2. Verify expected_answer is correct
                    3. Set quality_score (1=bad, 5=excellent)
                    4. Set expert_validated=true when reviewed
                    5. Add expert_notes if needed
                    6. Delete pairs that are unusable
                """,
            },
            "qa_pairs": [asdict(qa) for qa in self.generated_pairs],
        }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.generated_pairs)} pairs to {filepath}")
    
    @classmethod
    def import_validated(cls, filepath: str) -> List[GeneratedQA]:
        """Import expert-validated pairs."""
        with open(filepath) as f:
            data = json.load(f)
        
        pairs = []
        for item in data.get("qa_pairs", []):
            if item.get("expert_validated", False):
                pairs.append(GeneratedQA(**item))
        
        logger.info(f"Imported {len(pairs)} validated pairs")
        return pairs


# ─────────────────────────────────────────────────────────────────
# Conversion to Evaluation Format
# ─────────────────────────────────────────────────────────────────

def convert_to_eval_format(qa_pairs: List[GeneratedQA], output_path: str):
    """
    Convert validated Q&A pairs to the retrieval evaluation format.
    
    This creates the golden_retrieval.json format used by RetrievalEvaluator.
    """
    test_cases = []
    
    for qa in qa_pairs:
        if not qa.expert_validated:
            continue
        
        # Build tags from metadata
        tags = list(qa.tags)
        tags.append(qa.question_type)
        tags.append(qa.difficulty)
        if qa.country_specific:
            tags.append(qa.country_specific)
        
        test_case = {
            "query": qa.question,
            "expected_doc_ids": [qa.source_document],  # Would need doc_id mapping
            "expected_sections": [qa.source_section],
            "tags": tags,
            "description": f"Generated from {qa.source_document}",
            "expected_answer": qa.expected_answer,  # For answer quality eval
        }
        test_cases.append(test_case)
    
    output = {
        "description": "Generated and validated evaluation set",
        "version": "1.0",
        "test_cases": test_cases,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Converted {len(test_cases)} pairs to eval format: {output_path}")
