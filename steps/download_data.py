import os
import logging
from zenml import step
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.DEBUG)

@step
def download_dataset()-> DatasetDict:
    """
    Downloads the Plant Village dataset from the HuggingFace hub if the
    data is not already available. Altohug this dataset is public and does
    not require authentication, it is recommended to set up a HuggingFace
    token.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "dataset")
    logging.info("Checking for data...")

    if os.path.isdir(data_dir):
        logging.info(f"Dataset found on directory: {data_dir}")
        dataset = load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=data_dir)
    else:
        logging.info("Dataset directory was not found, downloading from the hub...")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.mkdir(data_dir)
        dataset = load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=data_dir)
        logging.info("Dataset downloaded successfully!")

    logging.info("Data retrieved successfully!")

    return dataset
