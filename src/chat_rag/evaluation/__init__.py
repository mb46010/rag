"""
Evaluation Module for HR RAG System

Provides tools to build and run evaluation sets:

1. **Synthetic Generation**: Auto-generate Q&A from docs
2. **Adversarial Cases**: Edge cases to stress-test the system
3. **Langfuse Integration**: Log traces, retroactive labeling
4. **Workflow Runner**: Orchestrate the full eval pipeline

Quick Start:
    # Generate eval set from docs
    python -m chat_rag.evaluation.runner --workflow generate
    
    # After manual review, import and run
    python -m chat_rag.evaluation.runner --workflow import
    
    # Or use programmatically
    from chat_rag.evaluation import EvalWorkflowRunner
    runner = EvalWorkflowRunner()
    await runner.run_full_workflow()
"""

from .synthetic_generation import (
    SyntheticQAGenerator,
    GeneratedQA,
    QuestionType,
    convert_to_eval_format,
)
from .adversarial_generation import (
    AdversarialGenerator,
    EdgeCase,
    EdgeCaseType,
    get_manual_hard_cases,
)
from .langfuse_integration import (
    LangfuseEvalIntegration,
    ProductionTrace,
    RelevanceLabel,
    AnswerQualityLabel,
    create_langfuse_callback,
)
from .runner import EvalWorkflowRunner

__all__ = [
    # Synthetic
    "SyntheticQAGenerator",
    "GeneratedQA",
    "QuestionType",
    "convert_to_eval_format",
    # Adversarial
    "AdversarialGenerator",
    "EdgeCase",
    "EdgeCaseType",
    "get_manual_hard_cases",
    # Langfuse
    "LangfuseEvalIntegration",
    "ProductionTrace",
    "RelevanceLabel",
    "AnswerQualityLabel",
    "create_langfuse_callback",
    # Runner
    "EvalWorkflowRunner",
]
