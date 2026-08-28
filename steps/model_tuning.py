import torch
import optuna
import mlflow
import logging
import datetime
import lightning as pl
from zenml import step
from optimization_loop import objective

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def hyperparameter_tuning(data_module, n_trials, n_epochs):
    logging.info("Initializing hyperparameter tuning process...")
    logging.info(f"The tuning process will have {n_trials} trials with {n_epochs} epochs each.")
    gpu_available = torch.cuda.is_available() # This is optional as one can create a parameter for this when defining the pipeline
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
