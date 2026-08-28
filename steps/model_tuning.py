import torch
import optuna
import mlflow
import logging
import datetime
import lightning as pl
from zenml import step
from typing import Dict, Any
from optimization_loop import objective

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def hyperparameter_tuning(data_module: pl.LightningDataModule, n_trials: int, n_epochs: int) -> Dict[str, Any]:
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
    gpu_available = torch.cuda.is_available() 
    accelerator = "gpu" if gpu_available else "cpu"
    n_jobs = 1 if gpu_available else -1

    date = datetime.date.today()
    with mlflow.start_run(run_name=f"{date} - Hyperparameter Tuning"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_epochs", n_epochs)

        active_run = mlflow.active_run()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, data_module, n_epochs, accelerator, active_run), 
            n_trials=n_trials,
            n_jobs=n_jobs
            )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_valf1_score", study.best_value)

        return study.best_params
    logging.info("The tuning process has ended successfully! Check the MLFlow UI to visualize results.")
