import torch
import mlflow
import logging
import numpy as np
from zenml import step
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from model_design.model_architecture import LightningClassifier

logging.basicConfig(level=logging.DEBUG)

@step(
    experiment_tracker="mlflow_tracker",
    settings={
        "experiment_tracker.mlflow":{
            "experiment_name": "model_deployment"
        }
    }
)
def promote_to_deployment(
    model: LightningClassifier,
    f1_score: float,
    deploy_config: dict
) -> None:
    """
    Promotes a model to the registry if the threshold is met.
    args:
        model: The model to be registered.
        f1_score: The model's F1 Score.
        deploy_config: The required configurations for the step.
    """
    if f1_score >= deploy_config["threshold"]:
        logging.info(f"The F1 Score ({f1_score:.4f}) meets the threshold: {deploy_config["threshold"]:.4f}!")
        logging.info("Promoting model...")

        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3, 128, 128))])

        example = torch.randn(1, 3, 128, 128)
        output = model.model(example)
        output_shape = tuple(output.shape)
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, *output_shape[1:]))])
        
        model_signature = ModelSignature(input_schema, output_schema)

        mlflow.pytorch.log_model(
            model.model,
            artifact_path="models",
            input_example=example,
            serialization_format="pt2",
            signature=model_signature
        )

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/models"

        mlflow.register_model(
            model_uri=model_uri,
            name=deploy_config["model_name"]
        )

        logging.info("Model has been successfully promoted to the resgistry!")

    else:
        logging.info(f"Sorry, this model couldn't be promoted." \
        f"The F1 Score ({f1_score}) is lower than the threshold: {deploy_config["threshold"]:.4f}.")
