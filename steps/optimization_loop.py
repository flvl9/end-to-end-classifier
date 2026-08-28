import mlflow
import lightning as pl
from typing import Dict, Any
from optuna.trial import Trial
from lightning.pytorch.loggers import MLFlowLogger
from model_design.model_architecture import LightningClassifier
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback


def get_hyperparameters(trial: Trial) -> Dict[str, Any]:
    """
    Generates the set of hyperparameters for a trial.
    args:
        trial - The optuna trial.
    returns:
        A dictionary containing the hyperparameters.
    """
    train_batch_size = trial.suggest_int("train_bacth_size", 16, 128)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    conv_layers = trial.suggest_int("conv_layers", 1, 4)
    filters = [trial.suggest_int(f"n_filters_{i}", 32, 128) for i in range(conv_layers)]
    kernel_sizes = [trial.suggest_int(f"kernel_size_{i}", 3, 5) for i in range(conv_layers)]
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    fc_size = trial.suggest_int("fc_size", 64, 256)

    return {
        "train_batch_size": train_batch_size,
        "lr": lr,
        "conv_layers": conv_layers,
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "dropout": dropout,
        "fc_size": fc_size
    }

def objective(trial: Trial, dm: pl.LightningDataModule, n_epochs: int, accelerator: str) -> float:
    """
    Defines the objective function to be optimized with optuna.
    args:
        trial - The optuna trial.
        dm - The Lightning datamodule containing the data.
        n_epochs - The number of epochs of each trial.
        accelerator - The accelerator to be used for the optimization.
        active_run - The mlflow run for the tracking process.
    returns:
        f1_score - The validation f1 score for the trial.
    """
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        hparams = get_hyperparameters(trial)

        mlflow.log_params(hparams)
        mlflow.log_param("accelerator", accelerator)

        classifier = LightningClassifier(
            conv_layers=hparams["conv_layers"],
            filters=hparams["filters"],
            kernel_sizes=hparams["kernel_sizes"],
            dropout=hparams["dropout"],
            fc_size=hparams["fc_size"],
            lr=hparams["lr"],
        )

        mlf_logger = MLFlowLogger()

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1score")],
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=mlf_logger,
            accelerator=accelerator,
            devices=1,
            enable_autolog_hparams=False

        )

        trainer.fit(model=classifier, datamodule=dm)


        if "val_f1score" in trainer.callback_metrics:
            f1_score = trainer.callback_metrics["val_f1score"].item()
            mlflow.log_metric("val_f1score", f1_score)
            return f1_score

        return 0.0
