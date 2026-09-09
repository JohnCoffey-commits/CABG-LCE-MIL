#!/usr/bin/env python3

import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model_path = args.repo_root / "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
    trainer_path = args.repo_root / "qwen-vl-finetune/qwenvl/train/anomaly_evidence_trainer.py"
    model_source = model_path.read_text(encoding="utf-8")
    trainer_source = trainer_path.read_text(encoding="utf-8")
    ast.parse(model_source)
    trainer_tree = ast.parse(trainer_source)
    if "last_evidence" in model_source or "last_evidence" in trainer_source:
        raise AssertionError("Forbidden stale last_evidence state is present.")
    compute_loss = next(
        node
        for node in ast.walk(trainer_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "compute_loss"
    )
    model_calls = [
        node
        for node in ast.walk(compute_loss)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "model"
    ]
    if len(model_calls) != 1:
        raise AssertionError(f"AnomalyEvidenceTrainer compute_loss has {len(model_calls)} model forwards.")
    required_fragments = (
        "pre_gate_evidence = A_log - N_log",
        "return_evidence: bool = False",
        "return_anomaly_evidence: bool = False",
        "anomaly_evidence: Optional[torch.FloatTensor] = None",
    )
    missing = [fragment for fragment in required_fragments if fragment not in model_source]
    if missing:
        raise AssertionError(f"Model evidence-output contract fragments are missing: {missing}")
    payload = {
        "status": "SUCCESS",
        "model_ast_valid": True,
        "trainer_ast_valid": True,
        "training_model_forward_calls": len(model_calls),
        "no_last_evidence_state": True,
        "explicit_opt_in_evidence_output": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
