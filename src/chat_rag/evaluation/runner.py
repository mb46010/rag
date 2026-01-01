"""
Evaluation Workflow Runner

Orchestrates the full evaluation workflow:
1. Generate synthetic Q&A (if no eval set exists)
2. Generate adversarial cases
3. Run retrieval evaluation
4. Log results to Langfuse
5. Generate report

Usage:
    # Full workflow
    python -m chat_rag.evaluation.runner --workflow full
    
    # Just generate eval set
    python -m chat_rag.evaluation.runner --workflow generate
    
    # Just run evaluation
    python -m chat_rag.evaluation.runner --workflow evaluate
"""

import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .synthetic_generation import SyntheticQAGenerator, convert_to_eval_format
from .adversarial_generation import AdversarialGenerator, get_manual_hard_cases
from .langfuse_integration import LangfuseEvalIntegration

logger = logging.getLogger(__name__)


class EvalWorkflowRunner:
    """
    Runs the complete evaluation workflow.
    
    Workflow steps:
    1. GENERATE: Create synthetic + adversarial eval cases
    2. REVIEW: Export for expert review (manual step)
    3. IMPORT: Load validated cases
    4. EVALUATE: Run retrieval evaluation
    5. REPORT: Generate report and log to Langfuse
    """
    
    def __init__(
        self,
        docs_dir: str = "documents",
        output_dir: str = "evaluation_output",
        policy_names: list = None,
    ):
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default policy names (customize for your docs)
        self.policy_names = policy_names or [
            "PTO", "WFH", "Expenses", "Parental Leave", "Sick Leave"
        ]
        
        # Initialize components
        self.synthetic_gen = SyntheticQAGenerator()
        self.adversarial_gen = AdversarialGenerator()
        self.langfuse = LangfuseEvalIntegration()
    
    async def run_generation_workflow(self) -> Dict[str, Any]:
        """
        Step 1: Generate evaluation cases.
        
        Creates:
        - synthetic_draft.json: Synthetic Q&A for review
        - adversarial_cases.json: Edge cases for review
        """
        results = {"generated": {}}
        
        # Generate synthetic Q&A
        logger.info("Generating synthetic Q&A pairs...")
        synthetic_pairs = await self.synthetic_gen.generate_from_documents(
            str(self.docs_dir),
            questions_per_section=3,
        )
        
        synthetic_path = self.output_dir / "synthetic_draft.json"
        self.synthetic_gen.export_for_review(str(synthetic_path))
        results["generated"]["synthetic"] = len(synthetic_pairs)
        results["generated"]["synthetic_path"] = str(synthetic_path)
        
        # Generate adversarial cases
        logger.info("Generating adversarial cases...")
        template_cases = self.adversarial_gen.generate_from_templates(
            policy_names=self.policy_names
        )
        
        # Add manual hard cases
        manual_cases = get_manual_hard_cases()
        self.adversarial_gen.edge_cases.extend(manual_cases)
        
        adversarial_path = self.output_dir / "adversarial_cases.json"
        self.adversarial_gen.export(str(adversarial_path))
        results["generated"]["adversarial"] = len(self.adversarial_gen.edge_cases)
        results["generated"]["adversarial_path"] = str(adversarial_path)
        
        logger.info(f"Generation complete. Review files in {self.output_dir}")
        logger.info("After review, run with --workflow import")
        
        return results
    
    def run_import_workflow(
        self,
        validated_synthetic_path: str = None,
        validated_adversarial_path: str = None,
    ) -> Dict[str, Any]:
        """
        Step 2: Import validated cases and create final eval set.
        
        Expects manually reviewed files with expert_validated=true.
        """
        results = {"imported": {}}
        
        # Default paths
        synthetic_path = validated_synthetic_path or str(
            self.output_dir / "synthetic_validated.json"
        )
        adversarial_path = validated_adversarial_path or str(
            self.output_dir / "adversarial_validated.json"
        )
        
        # Import validated synthetic
        if Path(synthetic_path).exists():
            validated_pairs = SyntheticQAGenerator.import_validated(synthetic_path)
            results["imported"]["synthetic"] = len(validated_pairs)
            
            # Convert to eval format
            eval_path = self.output_dir / "golden_synthetic.json"
            convert_to_eval_format(validated_pairs, str(eval_path))
            results["imported"]["synthetic_eval_path"] = str(eval_path)
        
        # Import adversarial (these don't need same conversion)
        if Path(adversarial_path).exists():
            # Adversarial cases are already in usable format
            results["imported"]["adversarial_path"] = adversarial_path
        
        logger.info(f"Import complete. Eval sets ready in {self.output_dir}")
        return results
    
    def run_evaluation_workflow(
        self,
        retriever,  # Your retriever instance
        eval_set_path: str = None,
        config_name: str = "default",
    ) -> Dict[str, Any]:
        """
        Step 3: Run retrieval evaluation.
        
        Args:
            retriever: Your EnhancedHybridRetriever instance
            eval_set_path: Path to evaluation set JSON
            config_name: Name for this config (for comparison)
        """
        from tests.eval import RetrievalEvaluator
        
        # Default eval set path
        eval_path = eval_set_path or str(self.output_dir / "golden_synthetic.json")
        
        if not Path(eval_path).exists():
            logger.error(f"Eval set not found: {eval_path}")
            logger.info("Run with --workflow generate first")
            return {"error": "Eval set not found"}
        
        # Run evaluation
        logger.info(f"Running evaluation with config: {config_name}")
        evaluator = RetrievalEvaluator.from_file(retriever, Path(eval_path))
        metrics = evaluator.evaluate()
        
        results = {
            "config": config_name,
            "eval_set": eval_path,
            "metrics": metrics.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Log to Langfuse
        self.langfuse.log_eval_results_to_langfuse(
            eval_name=f"retrieval_{config_name}",
            metrics=metrics.to_dict(),
            config_snapshot={"config_name": config_name},
        )
        
        # Print summary
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {config_name}")
        print("="*60)
        print(metrics)
        print("="*60 + "\n")
        
        return results
    
    def run_comparison(
        self,
        retriever_configs: Dict[str, Any],  # {name: retriever_instance}
        eval_set_path: str = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple retriever configurations.
        
        Useful for A/B testing different settings.
        """
        results = {"comparisons": []}
        
        for config_name, retriever in retriever_configs.items():
            config_results = self.run_evaluation_workflow(
                retriever=retriever,
                eval_set_path=eval_set_path,
                config_name=config_name,
            )
            results["comparisons"].append(config_results)
        
        # Generate comparison report
        self._generate_comparison_report(results["comparisons"])
        
        return results
    
    def _generate_comparison_report(self, comparisons: list):
        """Generate a comparison report."""
        print("\n" + "="*80)
        print("CONFIGURATION COMPARISON")
        print("="*80)
        
        # Header
        print(f"{'Config':<20} {'Recall@1':<10} {'Recall@3':<10} {'MRR':<10}")
        print("-"*50)
        
        for comp in comparisons:
            if "error" in comp:
                continue
            m = comp["metrics"]
            print(f"{comp['config']:<20} {m['recall_at_1']:<10.3f} {m['recall_at_3']:<10.3f} {m['mrr']:<10.3f}")
        
        print("="*80 + "\n")
    
    async def run_full_workflow(self, retriever=None):
        """Run the complete workflow."""
        logger.info("Starting full evaluation workflow...")
        
        # Step 1: Generate
        gen_results = await self.run_generation_workflow()
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Synthetic Q&A: {gen_results['generated']['synthetic']} pairs")
        print(f"Adversarial: {gen_results['generated']['adversarial']} cases")
        print("\nNEXT STEPS:")
        print("1. Review and validate: synthetic_draft.json")
        print("2. Review and validate: adversarial_cases.json")
        print("3. Save validated versions as *_validated.json")
        print("4. Run: python -m chat_rag.evaluation.runner --workflow import")
        print("="*60 + "\n")
        
        return gen_results


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Evaluation Workflow Runner")
    parser.add_argument(
        "--workflow",
        choices=["generate", "import", "evaluate", "full"],
        default="generate",
        help="Which workflow step to run",
    )
    parser.add_argument(
        "--docs-dir",
        default="documents",
        help="Directory containing policy documents",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_output",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--eval-set",
        help="Path to evaluation set (for evaluate workflow)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    runner = EvalWorkflowRunner(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
    )
    
    if args.workflow == "generate":
        await runner.run_generation_workflow()
    
    elif args.workflow == "import":
        runner.run_import_workflow()
    
    elif args.workflow == "evaluate":
        # For evaluate, you need to provide a retriever
        # This is a placeholder - in practice, initialize your retriever
        print("To run evaluation, initialize your retriever and call:")
        print("  runner.run_evaluation_workflow(retriever, eval_set_path)")
    
    elif args.workflow == "full":
        await runner.run_full_workflow()


if __name__ == "__main__":
    asyncio.run(main())
