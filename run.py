import yaml
import argparse
import datetime
from pipelines import plant_disease_training_pipeline, deploy_model

def unit_fraction(value: str) -> float:
    """
    Ensures that the value is in range (0, 1].
    """
    float_value = float(value)
    if not 0.0 < float_value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"The value must be between 0 and 1, got {float_value}"
            )
    return float_value

def main() -> None:
    date = datetime.date.today()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train",
        action="store_true",
        help="Runs the training pipeline."
    )
    ap.add_argument(
        "--deploy",
        action="store_true",
        help="Deploys a model based on a threshold."
    )
    ap.add_argument(
        "--run-id",
        type=str,
        help="The training pipeline run id containing the desired model to be deployed.",
        required=True
    )
    ap.add_argument(
        "--deployment-threshold",
        type=unit_fraction,
        default=0.75,
        help="The F1 Score threshold that should be met to deploy a model. " \
        "Must be positive and between 0 and 1."
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default=f"plantvillage_cnn_{date}",
        help="The name for the deployed model."
    )
    ap.add_argument(
        "--model-traffic",
        type=unit_fraction,
        help="The fraction (between 0 and 1) of requests to be sent to the model.",
        required=True
    )

    args = ap.parse_args()

    if args.train:
        print("Initializing training pipeline...")

        with open("configs/training_pipeline.yaml", "r") as file:
            configs = yaml.safe_load(file)

        run = plant_disease_training_pipeline(
            configs["datamodule"],
            configs["tuning"],
            configs["training"]
        )

        run_id = run.id
        print("The training pipeline has ended.")
        print(f"\nIf you would like to deploy the trained model, please use {run_id} as run id when running the deployment pipeline.")

    elif args.deploy:
        print("Initializing deployment pipeline...")
        run = deploy_model(
            args.run_id,
            args.deployment_threshold,
            args.model_name,
            args.model_traffic
        )
        run_id = run.id
        print(f"\n The deployment pipeline has ended.")
        print(f"\n Check Zenml's UI to see the run ({run_id}) details.")

    else:
        print("""Plant Village CNN End-to-End Pipeline

Usage:
    python run.py --train               # Trains a custom CNN
    python run.py --deploy              # Promotes a model to MLflow model registry

The deployment pipeline (--deploy), takes as input the following parameters:
    Mandatory parameters:
        --run-id: str                   # The Zenml's pipeline run id containig the desired model to be deployed (it can be found on the Zenml UI)
        --model-traffic: float          # The fraction of requests that will be assign to the model
    Optional parameters:
        --deployment-threshold: float   # The minimum F1 Score to deploy the model (0.75 by default). Must be between 0 and 1.
        --model-name: str               # The name that will be assigned to the model on the registry

Workflow:
1. Train a model: python run.py --train 
2. Deploy a model: python run.py --deploy --run-id {run_id} --model-traffic {model_traffic} [optional parameters]
3. Access the app. If working locally: streamlit run app/app_interface.py

To modify the training parameters, check the file configs/training_pipeline.yaml
""")

if __name__ == "__main__":
    main()
