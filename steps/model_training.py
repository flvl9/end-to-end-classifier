import mlflow
import logging
import lightning as pl
from zenml import step
from pathlib import Path
from typing import Dict, Any
from lightning.pytorch.loggers import MLFlowLogger
from model_design.model_architecture import LightningClassifier
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(
    data_module: pl.LightningDataModule,
    hparams: Dict[str, Any], 
    config: dict
    )-> Path:
    """
    Trains the model with the best hyperparameters from the hyperparameter tuning stage.
    args:
        data_module - A Lightning datamodule with the training and testing data.
        hparams - A dictionary containing the best hyperparameters.
        config: A dictionary containing the configurations of this step.
    returns:
        The path to the best model.
    """
    logging.info("Initializing model training process...")
    pl.seed_everything(seed=config["seed"], workers=True)

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

    mlf_logger = MLFlowLogger(log_model=True)

    early_stopping_callback = EarlyStopping(
        monitor="val_f1score",
        patience=config["early_stopping"]["patience"],
        mode='max',
        min_delta=config["early_stopping"]["min_delta"]
        )

    checkpoints = ModelCheckpoint(
        monitor="val_f1score",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/"
    )

    trainer = pl.Trainer(
        max_epochs=config["n_epochs"],
        callbacks=[early_stopping_callback, checkpoints],
        logger=mlf_logger,
        accelerator=config["accelerator"],
        devices=config["n_devices"]
        )

    trainer.fit(model=model, datamodule=data_module)

    if "val_f1score" in trainer.callback_metrics:
        f1_score = trainer.callback_metrics["val_f1score"].item()
        mlflow.log_metric("final_val_f1score", f1_score)

    logging.info("Model was trained and logged successfully!")

    return Path(checkpoints.best_model_path)
