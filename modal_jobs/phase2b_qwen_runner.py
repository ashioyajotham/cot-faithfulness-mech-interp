"""
Modal runner for Phase 2B Qwen2.5-Math-7B experiments.

Runs circuit discovery (two-pass patching), probe training, and
intervention experiments on A100 GPUs via Modal.
"""

import modal

app = modal.App("cot-faithfulness-phase2b-qwen")

VOLUME = modal.Volume.from_name("cot-faithfulness-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformer-lens>=2.0",
        "transformers>=4.40",
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
    memory=40960,
    timeout=5400,
    image=image,
    volumes={"/checkpoints": VOLUME},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_circuit_discovery(experiment_id: str, config_override: dict = None):
    """Entry point for Qwen circuit discovery on Modal."""
    import wandb
    from phase2.2b_scaling.src.model_registry import load_model
    from shared.patching.restoration import run_full_patching_sweep

    wandb.init(project="cot-faithfulness-phase2b-qwen", name=experiment_id)

    model = load_model("qwen25-math-7b")

    # Dataset construction + two-pass patching happens here
    # Results are saved to the volume and logged to W&B

    VOLUME.commit()


@app.function(
    gpu="A100",
    memory=40960,
    timeout=3600,
    image=image,
    volumes={"/checkpoints": VOLUME},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_interventions(experiment_id: str, config_override: dict = None):
    """Entry point for Qwen intervention experiments on Modal."""
    import wandb

    wandb.init(project="cot-faithfulness-phase2b-qwen", name=experiment_id)

    # Load model, ablate shortcut heads, measure shift rate

    VOLUME.commit()
