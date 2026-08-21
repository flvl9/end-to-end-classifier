import logging
from zenml import step
import lightning as pl
from datamodule.datamodule import PlantVillageDatamodule

logging.basicConfig(level=logging.DEBUG)

@step(enable_cache=False)
def get_datamodule(cache_dir: str, data_dir: str, train_batch_size: int, val_batch_size: int) -> pl.LightningDataModule:
    logging.info("Initializing data preparation...")
    data_module = PlantVillageDatamodule(cache_path=cache_dir,
                                         data_path=data_dir,
                                         train_batch_size=train_batch_size,
                                         val_batch_size=val_batch_size)
    logging.info("Datamodule was generated successfully!")

    return data_module
