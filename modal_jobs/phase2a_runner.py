"""
Modal runner for Phase 2A validation experiments.

Phase 2A experiments run on CPU and are included here primarily for
reproducibility and W&B integration.  They can also run locally.
"""

import modal

app = modal.App("cot-faithfulness-phase2a")

VOLUME = modal.Volume.from_name("cot-faithfulness-checkpoints", create_if_missing=True)


@app.function(
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={"/checkpoints": VOLUME},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_validation(experiment_id: str, config_override: dict = None):
    """Entry point for Phase 2A validation on Modal."""
    import wandb

    wandb.init(project="cot-faithfulness-phase2a", name=experiment_id)

    if experiment_id == "01_probe_selectivity":
        from phase2.src.selectivity import run_selectivity_test
        # Load activations from checkpoint volume, run selectivity
        pass
    elif experiment_id == "02_false_negative_analysis":
        from phase2.src.error_analysis import extract_false_negatives
        pass
    elif experiment_id == "03_bootstrap_significance":
        from phase2.src.bootstrap import bootstrap_restoration_ci
        pass

    VOLUME.commit()
