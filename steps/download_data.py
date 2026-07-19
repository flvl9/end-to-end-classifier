import os
from datasets import load_dataset

def download_dataset() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "dataset")

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        dataset = load_dataset("GVJahnavi/PlantVillage_dataset", split="train")
        dataset.save_to_disk(dataset_path=data_dir)
