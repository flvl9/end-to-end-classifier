import os
import torch
import mlflow
import lightning as pl
from typing import Dict, Any
from optuna.trial import Trial
from mlflow.tracking import MlflowClient
from lightning.pytorch.loggers import MLFlowLogger
from model_design.model_architecture import LightningClassifier
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def get_hyperparameters(trial: Trial, search_space: dict) -> Dict[str, Any]:
    """
    Generates the set of hyperparameters for a trial.
    args:
        trial - The optuna trial.
        search_space: A dictionary containing the search space for the hyperparameters.
    returns:
        A dictionary containing the hyperparameters.
    """
    conv_layers = trial.suggest_int("conv_layers",
                                  search_space["conv_layers"][0],
                                  search_space["conv_layers"][1])
    filters = [trial.suggest_int(f"n_filters_{i}",
                                  search_space["filters"][0],
                                  search_space["filters"][1]) 
                                  for i in range(conv_layers)]
    kernel_sizes = [trial.suggest_int(f"kernel_size_{i}",
                                  search_space["kernel_sizes"][0],
                                  search_space["kernel_sizes"][1])
                                  for i in range(conv_layers)]
    dropout = trial.suggest_float("dropout",
                                  search_space["dropout"][0],
                                  search_space["dropout"][1])
    fc_size = trial.suggest_int("fc_size",
                                  search_space["fc_size"][0],
                                  search_space["fc_size"][1])
    lr = trial.suggest_loguniform("lr",
                                  search_space["lr"][0],
                                  search_space["lr"][1])

    return {
        "conv_layers": conv_layers,
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "dropout": dropout,
        "fc_size": fc_size,
        "lr": lr,
    }

def objective(
        trial: Trial,
        dm: pl.LightningDataModule,
        n_epochs: int,
        accelerator: str,
        parent_run_id,
        search_space: dict,
        devices: int,
        class_weights: torch.Tensor,
        num_classes: int
        )-> float:
    """
    Defines the objective function to be optimized with optuna.
    args:
        trial - The optuna trial.
        dm - The Lightning datamodule containing the data.
        n_epochs - The number of epochs of each trial.
        accelerator - The accelerator to be used for the optimization.
        parent_run_id - The current parent run id.
        search_space: A dictionary containing the search space for the hyperparameters.
        devices: Number of devices for the lightning trainer.
        class_weights - The class weights to be applied.
        num_classes - The number of classes of the datamodule
    returns:
        f1_score - The validation f1 score for the trial.
    """
    client = MlflowClient()

    with mlflow.start_run(
        nested=True,
        run_name=f"trial_{trial.number}",
        parent_run_id=parent_run_id,
        experiment_id=mlflow.active_run().info.experiment_id
        ) as child_run:

        hparams = get_hyperparameters(trial, search_space)

        for k, v in hparams.items():
            client.log_param(child_run.info.run_id, k, str(v))
        client.log_param(child_run.info.run_id, "accelerator", accelerator)   

        classifier = LightningClassifier(
            **hparams,
            class_weights=class_weights,
            num_classes=num_classes
            )

        mlf_logger = MLFlowLogger(
            run_id=child_run.info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1score")],
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=mlf_logger,
            accelerator=accelerator,
            devices=devices
        )

        trainer.fit(model=classifier, datamodule=dm)

        if "val_f1score" in trainer.callback_metrics:
            f1_score = trainer.callback_metrics["val_f1score"].item()
            client.log_metric(child_run.info.run_id, "val_f1score", f1_score)
            return f1_score

        return 0.0
