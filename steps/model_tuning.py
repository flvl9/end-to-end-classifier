import torch
import optuna
import mlflow
import logging
import lightning as pl
from zenml import step
from typing import Dict, Any
from optimization_loop import objective

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def hyperparameter_tuning(data_module: pl.LightningDataModule, n_trials: int, n_epochs: int, seed: int = 32) -> Dict[str, Any]:
    """
    Defines the logic of the hyperparameter tuning process.
    The experiments, metrics and parameters are logged to mlflow.
    args:
        data_module - The Lightning datamodule containing the training/validation data.
        n_trials - The number of trials to perform the optimization process.
        n_epochs - The number of epochs that each trial will run.
    returns:
        A dictionary containing the best hyperparameters for further training the model.
    """
    logging.info("Initializing hyperparameter tuning process...")
    logging.info(f"The tuning process will have {n_trials} trials with {n_epochs} epochs each.")
    pl.seed_everything(seed=seed, workers=True)

    gpu_available = torch.cuda.is_available() 
    accelerator = "gpu" if gpu_available else "cpu"
    n_jobs = 1 if gpu_available else -1

    mlflow.log_param("n_trials", n_trials)
    mlflow.log_param("n_epochs", n_epochs)
    mlflow.log_param("seed", seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda trial: objective(trial, data_module, n_epochs, accelerator), 
        n_trials=n_trials,
        n_jobs=n_jobs
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_valf1_score", study.best_value)

    logging.info("The tuning process has ended successfully! Check the MLFlow UI to visualize results.")

    return study.best_params
