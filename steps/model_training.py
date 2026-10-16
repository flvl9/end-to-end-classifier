import torch
import mlflow
import logging
import lightning as pl
from datasets import DatasetDict
from typing import Annotated, Tuple
from zenml.enums import ArtifactType
from zenml import step, ArtifactConfig
from lightning.pytorch.loggers import MLFlowLogger
from datamodule.datamodule import PlantVillageDatamodule
from model_design.model_architecture import LightningClassifier
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from zenml.integrations.pytorch.materializers.pytorch_module_materializer import PyTorchModuleMaterializer

logging.basicConfig(level=logging.DEBUG)

@step(
        enable_cache=False,
        output_materializers={"final_model": PyTorchModuleMaterializer},
        settings={
            "experiment_tracker.mlflow":{
                "experiment_name": "model_training"
            }
        }
    )
def train_model(
    dataset: DatasetDict,
    best_model: LightningClassifier,
    best_params: dict,
    datamodule_config: dict,
    train_config: dict,
    ) -> Tuple[
        Annotated[
            LightningClassifier,
            ArtifactConfig(name="final_model", 
                           artifact_type=ArtifactType.MODEL)
        ],
        float
    ]:
    """
    Trains the model with the best hyperparameters from the hyperparameter tuning stage.
    args:
        dataset - A DatasetDict containg the training/validation data.
        best_model - The model with the best performance dunring the tuning process.
        best_params - A dictionary containg the best hyperparameters from the tuning process.
        datamodule_config - A dictionary containing the required parameters for the dm.
        train_config - A dictionary containing the required parameters for the step.
    returns:
        best_model - The final model.
        best_score - The validation f1 score for the best model.
    """
    logging.info("Initializing model training process...")
    pl.seed_everything(seed=train_config["seed"], workers=True)

    mlflow.log_params(best_params)
    mlflow.log_param("n_epochs_train", train_config["n_epochs"])
    mlflow.log_param("accelerator", train_config["accelerator"])
    mlflow.log_param("seed", train_config["seed"])
    mlflow.log_param("n_devices", train_config["n_devices"])
    mlflow.log_param("early_stopping_patience", train_config["early_stopping"]["patience"])
    mlflow.log_param("min_delta", train_config["early_stopping"]["min_delta"])

    # This call is necessary since the class weights depend on it.
    data_module = PlantVillageDatamodule(
        dataset=dataset,
        **datamodule_config
    )
    data_module.setup("fit")

    mlf_logger = MLFlowLogger(
        run_id=mlflow.active_run().info.run_id,
        tracking_uri=mlflow.get_tracking_uri()
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_f1score",
        patience=train_config["early_stopping"]["patience"],
        mode="max",
        min_delta=train_config["early_stopping"]["min_delta"]
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
        max_epochs=train_config["n_epochs"],
        callbacks=[early_stopping_callback, checkpoints],
        logger=mlf_logger,
        accelerator=train_config["accelerator"],
        devices=train_config["n_devices"],
        precision="bf16-mixed"
        )

    trainer.fit(
        model=best_model,
        datamodule=data_module
        )

    if checkpoints.best_model_score is None:
        raise RuntimeError("No checkpoint was saved. Vlidation likely never ran.")

    best_score = float(checkpoints.best_model_score)
    mlflow.log_metric("final_val_f1score", best_score)

    final_model_path = checkpoints.best_model_path
    ckpt = torch.load(final_model_path, weights_only=False)
    best_model.load_state_dict(ckpt["state_dict"])

    logging.info("Model was trained and logged successfully!")

    return best_model, best_score
