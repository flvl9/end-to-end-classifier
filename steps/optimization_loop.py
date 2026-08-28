import mlflow
import lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
from model_design.model_architecture import LightningClassifier
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback


def get_hyperparameters(trial):
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

def objective(trial, dm, n_epochs, accelerator, active_run):
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

            classifier._log_hyperparams = False
            mlf_logger = MLFlowLogger(
                tracking_uri=mlflow.get_tracking_uri(),
                run_id=active_run.info.run_id
                )

            trainer = pl.Trainer(
                max_epochs=n_epochs,
                callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1score")],
                enable_progress_bar=False,
                logger=mlf_logger,
                accelerator=accelerator,
                devices=1,

            )

            trainer.fit(model=classifier, datamodule=dm)

            f1_score = trainer.callback_metrics["val_f1score"].item()
            mlflow.log_metric("val_f1score", f1_score)

            return f1_score
       