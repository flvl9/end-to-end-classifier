import yaml
from zenml import pipeline
from steps import get_dataset, hyperparameter_tuning, train_model

@pipeline(name="plant_disease_training")
def plant_disease_training_pipeline(
    dm_config: dict,
    tuning_config: dict,
    training_config: dict
    ):
    dataset = get_dataset()
    hparams, model = hyperparameter_tuning(
        dataset,
        dm_config,
        tuning_config
    )
    final_model, val_f1 = train_model(
        dataset,
        model,
        hparams,
        dm_config,
        training_config
    )

    return final_model, val_f1
