from pathlib import Path
from typing import Optional
from src.ms2.training.baseline import train_baseline_run, RunConfig

def train_quick_ms2(
    model_type: str, 
    repo_root: Optional[Path] = None, 
    budget_minutes: float = 1.0,
    seed: int = 42
) -> str:
    """
    Thin wrapper around MS2 baseline training.
    Returns path to the best checkpoint.
    """
    repo_root = (repo_root or Path.cwd()).resolve()
    model_token = "a" if model_type.lower() == "model_a" else "b"
    
    config = RunConfig(
        model_id=model_token.upper(),
        seed=seed,
        wall_clock_budget_minutes=budget_minutes,
    )
    
    print(f"Starting quick train for {model_type} (budget: {budget_minutes} min)...")
    train_baseline_run(repo_root=repo_root, model_token=model_token, seed=seed, run_config=config)
    
    ckpt_path = repo_root / "experiments" / "ms2" / f"model_{model_token}" / str(seed) / "checkpoints" / "best.weights.h5"
    return str(ckpt_path)
