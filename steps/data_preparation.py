import os
import logging
from zenml import step
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import HFDatasetMaterializer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logging.basicConfig(level=logging.DEBUG)

@step(enable_cache=False,
      output_materializers=HFDatasetMaterializer)
def get_dataset() -> DatasetDict:
    """
    Retrieves the Plant Village dataset for model training.
    returns:
        A DatasetDict containing the training and validation data.
    """
    logging.info("Initializing data preparation...")

    return load_dataset("GVJahnavi/PlantVillage_dataset", token=HF_TOKEN)
