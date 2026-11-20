from zenml import pipeline
from zenml.client import Client
from steps import promote_to_deployment

@pipeline(name="plant_disease_model_deployment")
def deploy_model(
    run_id: str,
    threshold: float,
    model_name: str,
    traffic: float
) -> None:
    run = Client().get_pipeline_run(run_id)
    train_pipeline_outputs = run.steps["train_model"].outputs
    val_f1, model = train_pipeline_outputs["output_1"][0], train_pipeline_outputs["final_model"][0]
    promote_to_deployment(
        model,
        val_f1,
        threshold,
        model_name,
        traffic
    )
