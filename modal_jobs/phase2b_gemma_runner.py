"""
Modal runner for Phase 2B Gemma 3 12B IT experiments.

Runs circuit discovery and SAE-based interpretation on A100 GPUs.
"""

import modal

app = modal.App("cot-faithfulness-phase2b-gemma")

VOLUME = modal.Volume.from_name("cot-faithfulness-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformer-lens>=2.0",
        "transformers>=4.40",
        "sae-lens",
        "scikit-learn>=1.4",
        "scipy>=1.12",
        "einops",
        "jaxtyping",
        "wandb>=0.16",
        "numpy>=1.26",
    )
)


@app.function(
    gpu="A100",
    memory=81920,
    timeout=7200,
    image=image,
    volumes={"/checkpoints": VOLUME},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_gemma_pipeline(experiment_id: str, config_override: dict = None):
    """Entry point for Gemma 3 circuit discovery + SAE analysis."""
    import wandb
    from phase2.2b_scaling.src.model_registry import load_model

    wandb.init(project="cot-faithfulness-phase2b-gemma", name=experiment_id)

    model = load_model("gemma3-12b-it")

    # Circuit discovery + SAE interpretation happens here

    VOLUME.commit()
