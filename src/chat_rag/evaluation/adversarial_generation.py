"""
Adversarial & Edge Case Generator

Generates challenging test cases that stress-test the RAG system:
- Cross-policy questions
- Negations
- Temporal queries
- Ambiguous references
- Out-of-scope detection

These cases find real weaknesses that synthetic generation misses.
"""

import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class EdgeCaseType(Enum):
    """Categories of adversarial/edge cases."""
    
    # Cross-document reasoning
    CROSS_POLICY = "cross_policy"           # "How does WFH affect expense claims?"
    CONTRADICTION = "contradiction"          # When policies might conflict
    
    # Linguistic challenges
    NEGATION = "negation"                   # "What is NOT covered?"
    DOUBLE_NEGATIVE = "double_negative"     # "Is it not allowed to not report...?"
    IMPLICIT = "implicit"                   # Answer requires inference
    
    # Temporal
    TEMPORAL_CURRENT = "temporal_current"   # "What's the current policy?"
    TEMPORAL_CHANGE = "temporal_change"     # "What changed in 2024?"
    TEMPORAL_FUTURE = "temporal_future"     # "Will this change next year?"
    
    # Scope testing
    OUT_OF_SCOPE = "out_of_scope"          # Should say "I don't know"
    PARTIAL_SCOPE = "partial_scope"         # Partially answerable
    
    # Reference challenges  
    AMBIGUOUS_REF = "ambiguous_ref"        # "What about the other policy?"
    PRONOUN_HEAVY = "pronoun_heavy"        # "Can I do it there?"
    
    # Multi-part
    MULTI_PART = "multi_part"              # "What's X and how does Y work?"
    COMPARISON = "comparison"               # "What's the difference between X and Y?"
    
    # Country-specific
    COUNTRY_CONTRAST = "country_contrast"   # "How is this different in Italy vs Switzerland?"
    COUNTRY_MISSING = "country_missing"     # Country-specific Q without specifying country


@dataclass
class EdgeCase:
    """An adversarial test case."""
    question: str
    edge_case_type: str
    expected_behavior: str  # What should the system do?
    expected_docs: List[str] = field(default_factory=list)
    should_answer: bool = True  # False for out-of-scope
    should_clarify: bool = False  # True if should ask for clarification
    difficulty: str = "hard"
    notes: str = ""
    tags: List[str] = field(default_factory=list)


# Templates for generating edge cases
EDGE_CASE_TEMPLATES = {
    EdgeCaseType.CROSS_POLICY: [
        "How does {policy_a} interact with {policy_b}?",
        "If I'm using {policy_a}, can I also claim {policy_b}?",
        "What happens to my {policy_a} if I take {policy_b}?",
    ],
    EdgeCaseType.NEGATION: [
        "What expenses are NOT reimbursable?",
        "When can I NOT work from home?",
        "What is NOT covered under {policy}?",
        "Are there any exceptions where {rule} doesn't apply?",
    ],
    EdgeCaseType.TEMPORAL_CHANGE: [
        "What changed in the {policy} this year?",
        "Is the {policy} different from last year?",
        "When did the {rule} come into effect?",
    ],
    EdgeCaseType.OUT_OF_SCOPE: [
        "What's the stock price today?",
        "Can you help me write a Python script?",
        "What's the weather in Zurich?",
        "Who is the CEO of Google?",
    ],
    EdgeCaseType.COUNTRY_CONTRAST: [
        "How is {policy} different between Switzerland and Italy?",
        "Do I get more {benefit} in CH or IT?",
        "Is the {process} the same in both countries?",
    ],
    EdgeCaseType.AMBIGUOUS_REF: [
        "What about the other option?",
        "Can you tell me more about that?",
        "How does the second one work?",
        "What's the deadline for it?",
    ],
    EdgeCaseType.MULTI_PART: [
        "What's the WFH policy and how do I request time off?",
        "How many PTO days do I have and what's the expense limit?",
        "Can I work remotely and claim home office expenses?",
    ],
}


class AdversarialGenerator:
    """
    Generates adversarial test cases to find RAG weaknesses.
    
    Two modes:
    1. Template-based: Uses predefined templates with policy names
    2. LLM-based: Generates creative edge cases from document analysis
    """
    
    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.8)
        self.edge_cases: List[EdgeCase] = []
    
    def generate_from_templates(
        self,
        policy_names: List[str],
        benefits: List[str] = None,
        rules: List[str] = None,
    ) -> List[EdgeCase]:
        """
        Generate edge cases using templates.
        
        Args:
            policy_names: List of policy names (e.g., ["PTO", "WFH", "Expenses"])
            benefits: Benefit types (e.g., ["vacation days", "sick leave"])
            rules: Rule descriptions (e.g., ["expense limits", "approval process"])
        """
        cases = []
        benefits = benefits or ["leave", "benefits", "allowance"]
        rules = rules or ["this policy", "the process", "the limit"]
        
        # Cross-policy combinations
        for i, policy_a in enumerate(policy_names):
            for policy_b in policy_names[i+1:]:
                for template in EDGE_CASE_TEMPLATES[EdgeCaseType.CROSS_POLICY]:
                    question = template.format(policy_a=policy_a, policy_b=policy_b)
                    cases.append(EdgeCase(
                        question=question,
                        edge_case_type=EdgeCaseType.CROSS_POLICY.value,
                        expected_behavior="Should retrieve from both policies and synthesize",
                        expected_docs=[policy_a, policy_b],
                        tags=["cross_policy", policy_a.lower(), policy_b.lower()],
                    ))
        
        # Negation cases
        for policy in policy_names:
            for template in EDGE_CASE_TEMPLATES[EdgeCaseType.NEGATION]:
                if "{policy}" in template:
                    question = template.format(policy=policy)
                else:
                    question = template
                cases.append(EdgeCase(
                    question=question,
                    edge_case_type=EdgeCaseType.NEGATION.value,
                    expected_behavior="Should find exclusions/exceptions in policy",
                    expected_docs=[policy] if "{policy}" in template else [],
                    tags=["negation"],
                ))
        
        # Out of scope (should NOT answer)
        for template in EDGE_CASE_TEMPLATES[EdgeCaseType.OUT_OF_SCOPE]:
            cases.append(EdgeCase(
                question=template,
                edge_case_type=EdgeCaseType.OUT_OF_SCOPE.value,
                expected_behavior="Should indicate this is outside HR scope",
                should_answer=False,
                tags=["out_of_scope"],
            ))
        
        # Ambiguous references (should clarify)
        for template in EDGE_CASE_TEMPLATES[EdgeCaseType.AMBIGUOUS_REF]:
            cases.append(EdgeCase(
                question=template,
                edge_case_type=EdgeCaseType.AMBIGUOUS_REF.value,
                expected_behavior="Should ask for clarification",
                should_answer=False,
                should_clarify=True,
                tags=["ambiguous", "clarification_needed"],
            ))
        
        # Country contrast
        for policy in policy_names:
            for template in EDGE_CASE_TEMPLATES[EdgeCaseType.COUNTRY_CONTRAST]:
                question = template.format(
                    policy=policy,
                    benefit=benefits[0] if benefits else "benefits",
                    process=rules[0] if rules else "process",
                )
                cases.append(EdgeCase(
                    question=question,
                    edge_case_type=EdgeCaseType.COUNTRY_CONTRAST.value,
                    expected_behavior="Should retrieve both country versions and compare",
                    expected_docs=[f"{policy}_CH", f"{policy}_IT"],
                    tags=["country_contrast", "comparison"],
                ))
        
        self.edge_cases.extend(cases)
        logger.info(f"Generated {len(cases)} template-based edge cases")
        return cases
    
    async def generate_creative_cases(
        self,
        documents_summary: str,
        n_cases: int = 10,
    ) -> List[EdgeCase]:
        """
        Use LLM to generate creative edge cases based on document analysis.
        
        Args:
            documents_summary: Summary of available policies
            n_cases: Number of cases to generate
        """
        prompt = f"""You are a QA engineer testing an HR chatbot. 
Your goal is to find edge cases that might break the system.

Available HR policies:
{documents_summary}

Generate {n_cases} tricky questions that would be hard for a RAG system to handle.

Focus on:
1. Questions requiring info from multiple policies
2. Questions about what's NOT allowed (negations)
3. Ambiguous questions that need clarification
4. Questions about policy changes over time
5. Questions that are slightly outside HR scope
6. Questions with pronouns/references that are unclear

For each, specify:
- question: The tricky question
- edge_case_type: cross_policy|negation|temporal|out_of_scope|ambiguous|multi_part
- expected_behavior: What should the system ideally do?
- should_answer: true/false
- should_clarify: true/false
- difficulty: hard/very_hard

Return as JSON array."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="Generate adversarial test cases for RAG evaluation."),
                HumanMessage(content=prompt),
            ])
            
            cases_data = self._parse_json_response(response.content)
            
            cases = []
            for item in cases_data:
                case = EdgeCase(
                    question=item["question"],
                    edge_case_type=item.get("edge_case_type", "unknown"),
                    expected_behavior=item.get("expected_behavior", ""),
                    should_answer=item.get("should_answer", True),
                    should_clarify=item.get("should_clarify", False),
                    difficulty=item.get("difficulty", "hard"),
                    tags=["llm_generated", item.get("edge_case_type", "unknown")],
                )
                cases.append(case)
            
            self.edge_cases.extend(cases)
            logger.info(f"Generated {len(cases)} creative edge cases")
            return cases
            
        except Exception as e:
            logger.error(f"Creative generation failed: {e}")
            return []
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Extract JSON from LLM response."""
        import re
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)
    
    def export(self, filepath: str):
        """Export edge cases to JSON."""
        data = {
            "description": "Adversarial and edge case test set",
            "total_cases": len(self.edge_cases),
            "by_type": self._count_by_type(),
            "cases": [asdict(c) for c in self.edge_cases],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.edge_cases)} edge cases to {filepath}")
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count cases by type."""
        counts = {}
        for case in self.edge_cases:
            counts[case.edge_case_type] = counts.get(case.edge_case_type, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────
# Manual "Known Hard" Cases
# ─────────────────────────────────────────────────────────────────

def get_manual_hard_cases() -> List[EdgeCase]:
    """
    Manually curated hard cases based on domain knowledge.
    
    ADD YOUR OWN KNOWN PAIN POINTS HERE.
    These are questions you know users ask that are tricky.
    """
    return [
        # ─── Add your known hard cases here ───
        
        EdgeCase(
            question="I'm going on maternity leave next month, what happens to my PTO?",
            edge_case_type="cross_policy",
            expected_behavior="Should explain PTO accrual during leave + maternity policy",
            expected_docs=["PTO", "Parental Leave"],
            tags=["cross_policy", "known_hard"],
            notes="Common question, requires synthesizing two policies",
        ),
        
        EdgeCase(
            question="My manager said I can't WFH on Fridays, is that allowed?",
            edge_case_type="implicit",
            expected_behavior="Should cite WFH policy re: manager discretion",
            expected_docs=["WFH"],
            tags=["implicit", "manager_discretion"],
            notes="Tests understanding of policy vs manager authority",
        ),
        
        EdgeCase(
            question="What's the maximum I can expense without receipts?",
            edge_case_type="negation",
            expected_behavior="Should find receipt requirements and any exceptions",
            expected_docs=["Expenses"],
            tags=["negation", "limits"],
        ),
        
        # Add more based on your actual user questions...
    ]
