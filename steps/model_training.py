import mlflow
import logging
import lightning as pl
from zenml import step
from typing import Dict, Any
from lightning.pytorch.loggers import MLFlowLogger
from model_design.model_architecture import LightningClassifier
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(
    data_module: pl.LightningDataModule,
    hparams: Dict[str, Any], 
    accelerator: str = "gpu", 
    n_epochs_train: int = 10,
    n_devices: int = 1,
    early_stopping_rounds: int = 3,
    delta: float = 0.0,
    seed: int = 32
    )-> str:
    """
    Trains the model with the best hyperparameters from the hyperparameter tuning stage.
    args:
        data_module - A Lightning datamodule with the training and testing data.
        hparams - A dictionary containing the best hyperparameters.
        accelerator - The device in which the training process will run.
        n_epochs_train - Number of epochs to train the model.
        n_devices - The number of devices or processes to run the training stage.
        early_stopping_rounds - The number of epochs to stop the training process if no progress is made.
        delta - The change of progress that shouldn't be surpassed to stop the model.
        seed - The seed to ensure experiment reproducibility and consistency.
    returns:
        A string containing the path to the best model.
    """
    logging.info("Initializaing model training process...")
    pl.seed_everything(seed=seed, workers=True)

    mlflow.log_params(hparams)
    mlflow.log_param("n_epochs_train", n_epochs_train)
    mlflow.log_param("accelerator", accelerator)
    mlflow.log_param("seed", seed)
    mlflow.log_param("n_devices", n_devices)
    mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
    mlflow.log_param("delta", delta)

    model = LightningClassifier(
        conv_layers=hparams["conv_layers"],
        filters=hparams["filters"],
        kernel_sizes=hparams["kernel_sizes"],
        dropout=hparams["dropout"],
        fc_size=hparams["fc_size"],
        lr=hparams["lr"]
        )

    mlf_logger = MLFlowLogger(log_model=True)

    early_stopping_callback = EarlyStopping(
        monitor="val_f1score",
        patience=early_stopping_rounds,
        mode='max',
        min_delta=delta
        )

    checkpoints = ModelCheckpoint(
        monitor="val_f1score",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/"
    )

    trainer = pl.Trainer(
        max_epochs=n_epochs_train,
        callbacks=[early_stopping_callback, checkpoints],
        logger=mlf_logger,
        accelerator=accelerator,
        devices=n_devices,
        enable_autolog_hparams=False
        )

    trainer.fit(model=model, datamodule=data_module)

    if "val_f1score" in trainer.callback_metrics:
        f1_score = trainer.callback_metrics["val_f1score"].item()
        mlflow.log_metric("final_val_f1score", f1_score)

    logging.info("Model was trained and logged successfully!")

    return checkpoints.best_model_path
