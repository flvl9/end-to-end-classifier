import torch
import optuna
import mlflow
import logging
import lightning as pl
from zenml import step
from typing import Dict, Any
from .optimization_loop import objective

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def hyperparameter_tuning(
    data_module: pl.LightningDataModule, 
    config: dict
    ) -> Dict[str, Any]:
    """
    Defines the logic of the hyperparameter tuning process.
    The experiments, metrics, and parameters are logged to mlflow.
    args:
        data_module - The Lightning datamodule containing the training/validation data.
        config - A dictionary containg the configurations for this step.
    returns:
        A dictionary containing the best hyperparameters for further training the model.
    """
    logging.info("Initializing hyperparameter tuning process...")
    logging.info(f"The tuning process will have {config["n_trials"]} trials with {config["n_epochs"]} epochs each.")
    pl.seed_everything(seed=config["seed"], workers=True)

    mlflow.log_param("n_trials", config["n_trials"])
    mlflow.log_param("n_epochs", config["n_epochs"])
    mlflow.log_param("seed", config["seed"])

    # This call is necessary since the class weights depend on it.
    data_module.setup("fit")
    class_weights = data_module.class_weights
    num_classes = data_module.num_classes

    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"])
    )
    study.optimize(
        lambda trial: objective(
            trial,
            data_module,
            config["n_epochs"],
            config["accelerator"],
            parent_run_id,
            config["search_space"],
            config["devices"],
            class_weights,
            num_classes
            ), 
        n_trials=config["n_trials"],
        n_jobs=config["optuna_n_jobs"]
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_valf1_score", study.best_value)

    logging.info("The tuning process has ended successfully! Check the MLFlow UI to visualize results.")

    return study.best_params
