import torch
import optuna
import mlflow
import logging
import lightning as pl
from datasets import DatasetDict
from typing import Tuple, Annotated
from zenml.enums import ArtifactType
from zenml import step, ArtifactConfig
from datamodule.datamodule import PlantVillageDatamodule
from model_design.model_architecture import LightningClassifier
from .optimization_loop import objective, get_best_model, format_hyperparameters
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import PyTorchModuleMaterializer

logging.basicConfig(level=logging.DEBUG)

@step(
        enable_cache=False,
        output_materializers={"best_model": PyTorchModuleMaterializer},
        settings={
            "experiment_tracker.mlflow":{
                "experiment_name": "tuning_process"
            }
        }
    )
def hyperparameter_tuning(
    dataset: DatasetDict,
    datamodule_config: dict,
    tuning_config: dict
    ) -> Tuple[
        Annotated[
            dict,
            ArtifactConfig(name="best_params", artifact_type=ArtifactType.DATA)],
        Annotated[
            LightningClassifier, 
            ArtifactConfig(name="best_model", artifact_type=ArtifactType.MODEL)]
    ]:
    """
    Performs hyperparameter tuning using optuna to find the best set of hyperparameters
    for the model.
    args:
        dataset - A DatasetDict containg the training/validation data.
        datamodule_config - A dictionary containing the required parameters for the dm.
        tuning_config - A dictionary containing the required parameters for the step.
    returns:
        The best model and its hyperparameters.
    """
    logging.info("Initializing hyperparameter tuning process...")
    logging.info(f"The tuning process will have {tuning_config["n_trials"]} trials with {tuning_config["n_epochs"]} epochs each.")
    pl.seed_everything(seed=tuning_config["seed"], workers=True)

    mlflow.log_param("n_trials", tuning_config["n_trials"])
    mlflow.log_param("n_epochs", tuning_config["n_epochs"])
    mlflow.log_param("seed", tuning_config["seed"])

    data_module = PlantVillageDatamodule(
        dataset=dataset,
        **datamodule_config
    )

    # This call is necessary since the class weights depend on it.
    data_module.setup("fit")
    class_weights = data_module.class_weights
    num_classes = data_module.num_classes

    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=tuning_config["seed"],
            n_startup_trials=10
            ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5
            )
    )
    study.optimize(
        lambda trial: objective(
            trial,
            data_module,
            tuning_config["n_epochs"],
            tuning_config["accelerator"],
            parent_run_id,
            tuning_config["search_space"],
            tuning_config["trainer_devices"],
            class_weights,
            num_classes
            ), 
        n_trials=tuning_config["n_trials"],
        n_jobs=tuning_config["optuna_n_jobs"], # <- If working with one GPU, set to 1 to prevent OOM errors.
        callbacks=[get_best_model]
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_valf1_score", study.best_value)

    logging.info(f"The tuning process has ended successfully with score: {study.best_value}! Check the MLFlow UI to visualize results.")

    best_ckpt_path = study.best_trial.user_attrs["best_path"]
    ckpt = torch.load(best_ckpt_path, weights_only=False)
    formatted_params = format_hyperparameters(study.best_params)
    model = LightningClassifier(
        **formatted_params,
        class_weights=class_weights,
        num_classes=num_classes
    )
    model.load_state_dict(ckpt["state_dict"])

    return formatted_params, model
