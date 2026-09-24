import logging
from zenml import step
from datamodule.datamodule import PlantVillageDatamodule
from materializers.datamodule_materializer import PlantVillageMaterializer

logging.basicConfig(level=logging.DEBUG)

@step(enable_cache=False,
      output_materializers=PlantVillageMaterializer)
def get_datamodule(config: dict) -> PlantVillageDatamodule:
    """
    Creates a LightningDataModule for the training process.
    args:
        config: A dictionary containing the configuration for the step.
    returns:
        data_module - The LightningDataModule object required for model training.
    """
    logging.info("Initializing data preparation...")
    data_module = PlantVillageDatamodule(**config)
    logging.info("Datamodule was generated successfully!")

    return data_module
