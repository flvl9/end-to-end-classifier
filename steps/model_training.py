import os
import torch
import shutil
import mlflow
import logging
import lightning as pl
from zenml.enums import ArtifactType
from zenml import step, ArtifactConfig
from typing import Dict, Any, Annotated, Tuple
from lightning.pytorch.loggers import MLFlowLogger
from .optimization_loop import format_hyperparameters
from model_design.model_architecture import LightningClassifier
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(
    data_module: pl.LightningDataModule,
    tuning_results: Dict[str, Any], 
    config: dict
    )-> Tuple[
        Annotated[str, ArtifactConfig(
            name="final_model",
            artifact_type=ArtifactType.MODEL)],
        float
        ]:
    """
    Trains the model with the best hyperparameters from the hyperparameter tuning stage.
    args:
        data_module - A Lightning datamodule with the training and testing data.
        tuning_results - A dictionary containing the results of the hyperparameter tuning process,
            (best hyperparameters, and the ckpt path to further train the best model).
        config: A dictionary containing the configurations of this step.
    returns:
        The path to the best model.
    """
    logging.info("Initializing model training process...")
    pl.seed_everything(seed=config["seed"], workers=True)

    hparams = format_hyperparameters(tuning_results["best_params"])

    mlflow.log_params(hparams)
    mlflow.log_param("n_epochs_train", config["n_epochs"])
    mlflow.log_param("accelerator", config["accelerator"])
    mlflow.log_param("seed", config["seed"])
    mlflow.log_param("n_devices", config["n_devices"])
    mlflow.log_param("early_stopping_patience", config["early_stopping"]["patience"])
    mlflow.log_param("min_delta", config["early_stopping"]["min_delta"])

    # This call is necessary since the class weights depend on it.
    data_module.setup("fit")

    model = LightningClassifier(
        **hparams,
        class_weights=data_module.class_weights,
        num_classes=data_module.num_classes
        )

    mlf_logger = MLFlowLogger(
        run_id=mlflow.active_run().info.run_id,
        tracking_uri=mlflow.get_tracking_uri()
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_f1score",
        patience=config["early_stopping"]["patience"],
        mode="max",
        min_delta=config["early_stopping"]["min_delta"]
        )

    checkpoints = ModelCheckpoint(
        monitor="val_f1score",
        mode="max",
        save_top_k=1,
        dirpath="train_checkpoints/",
        save_last=False,
        enable_version_counter=False
    )

    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        max_epochs=config["n_epochs"],
        callbacks=[early_stopping_callback, checkpoints],
        logger=mlf_logger,
        accelerator=config["accelerator"],
        devices=config["n_devices"],
        precision="bf16-mixed"
        )

    ckpt_path = tuning_results["best_ckpt_path"]
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    trainer.fit(
        model=model,
        datamodule=data_module
        )

    if checkpoints.best_model_score is None:
        raise RuntimeError("No checkpoint was saved. Vlidation likely never ran.")

    best_score = float(checkpoints.best_model_score)
    mlflow.log_metric("final_val_f1score", best_score)

    best_model_path = checkpoints.best_model_path

    os.makedirs("artifacts", exist_ok=True)
    shutil.copy(best_model_path, "artifacts/final.ckpt")
    shutil.rmtree("hp_tuning_checkpoint/", ignore_errors=True)
    shutil.rmtree("train_checkpoints/", ignore_errors=True)

    logging.info("Model was trained and logged successfully!")

    return "artifacts/final.ckpt", best_score
