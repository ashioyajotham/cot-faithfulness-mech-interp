"""Upload Phase 2B datasets to HuggingFace Hub.

Uploads the Qwen 1.5B and Qwen 7B contrastive pair datasets as separate
configs under the existing CoT_Faithfulness_Dataset repo.

Usage:
    # Set your HF token first
    export HF_TOKEN=hf_...

    # Upload both datasets
    python scripts/upload_datasets_hf.py

    # Upload only one model
    python scripts/upload_datasets_hf.py --model qwen25-math-1.5b
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict, Features, Value
    from huggingface_hub import login
except ImportError:
    print("Install required packages:")
    print("  pip install datasets huggingface-hub")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "phase2" / "2b_scaling" / "results"

REPO_ID = "ashioyajotham/CoT_Faithfulness_Dataset"

MODEL_CONFIGS = {
    "qwen25-math-1.5b": {
        "description": "Qwen2.5-1.5B-Instruct contrastive pairs (500 generated, 205 valid)",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_params": "1.5B",
    },
    "qwen25-math-7b": {
        "description": "Qwen2.5-7B-Instruct contrastive pairs (500 generated, 309 valid)",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_params": "7B",
    },
}

FEATURES = Features({
    "a": Value("int64"),
    "b": Value("int64"),
    "correct_answer": Value("int64"),
    "corrupted_answer": Value("int64"),
    "carry_required": Value("bool"),
    "corruption_type": Value("string"),
    "corruption_severity": Value("string"),
    "faithful_prompt": Value("string"),
    "unfaithful_prompt": Value("string"),
    "model_answer_faithful": Value("int64"),   # parsed answer, -1 if parse failed
    "model_answer_unfaithful": Value("int64"),  # parsed answer, -1 if parse failed
    "label": Value("int64"),  # 0=faithful, 1=unfaithful
    "is_valid": Value("bool"),
})


def load_dataset_json(model_key: str) -> list:
    """Load the dataset.json for a model."""
    path = RESULTS_BASE / model_key / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}")
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} pairs from {path}")
    return data


def prepare_splits(data: list) -> DatasetDict:
    """Split into 'all' (everything) and 'valid' (is_valid=True) subsets."""
    valid = [p for p in data if p.get("is_valid")]

    # Ensure consistent types
    for pairs in [data, valid]:
        for p in pairs:
            p["a"] = int(p["a"])
            p["b"] = int(p["b"])
            p["correct_answer"] = int(p["correct_answer"])
            p["corrupted_answer"] = int(p["corrupted_answer"])
            p["carry_required"] = bool(p.get("carry_required", False))
            p["corruption_type"] = str(p.get("corruption_type", ""))
            p["corruption_severity"] = str(p.get("corruption_severity", ""))
            p["model_answer_faithful"] = int(p.get("model_answer_faithful", -1))
            p["model_answer_unfaithful"] = int(p.get("model_answer_unfaithful", -1))
            p["label"] = int(p.get("label", -1))
            p["is_valid"] = bool(p.get("is_valid", False))

    ds_all = Dataset.from_list(data, features=FEATURES)
    ds_valid = Dataset.from_list(valid, features=FEATURES)

    return DatasetDict({
        "all": ds_all,
        "valid": ds_valid,
    })


def upload_model_dataset(model_key: str, dry_run: bool = False):
    """Upload a single model's dataset to HF Hub."""
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Uploading: {model_key}")
    print(f"  Model: {config['model_name']} ({config['model_params']})")
    print(f"{'='*60}")

    data = load_dataset_json(model_key)
    ds_dict = prepare_splits(data)

    n_all = len(ds_dict["all"])
    n_valid = len(ds_dict["valid"])
    labels = ds_dict["valid"]["label"]
    n_unfaithful = sum(1 for l in labels if l == 1)
    n_faithful = sum(1 for l in labels if l == 0)

    print(f"  All pairs:    {n_all}")
    print(f"  Valid pairs:  {n_valid}")
    print(f"  Unfaithful:   {n_unfaithful} ({100*n_unfaithful/n_valid:.0f}%)")
    print(f"  Faithful:     {n_faithful} ({100*n_faithful/n_valid:.0f}%)")

    if dry_run:
        print("  [DRY RUN] Would push to HF Hub")
        return

    # Push as a config (subset) of the main dataset
    config_name = model_key.replace("-", "_")
    ds_dict.push_to_hub(
        REPO_ID,
        config_name=config_name,
        commit_message=f"Add {model_key} dataset ({n_all} pairs, {n_valid} valid)",
    )
    print(f"  ✓ Pushed as config '{config_name}' to {REPO_ID}")


def main():
    parser = argparse.ArgumentParser(description="Upload Phase 2B datasets to HuggingFace")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        help="Upload only this model (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without pushing")
    args = parser.parse_args()

    # Authenticate
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("✓ Authenticated with HF Hub")
    elif not args.dry_run:
        print("ERROR: Set HF_TOKEN environment variable")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    models = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    for model_key in models:
        upload_model_dataset(model_key, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Dataset: https://huggingface.co/datasets/{REPO_ID}")
    print("  Load in Python:")
    for m in models:
        config_name = m.replace("-", "_")
        print(f"    ds = load_dataset('{REPO_ID}', '{config_name}', split='valid')")


if __name__ == "__main__":
    main()
