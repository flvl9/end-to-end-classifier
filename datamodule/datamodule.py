import os
import logging
import lightning as pl
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from datamodule_utils import train_transforms, val_transforms, collate

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logging.basicConfig(level=logging.DEBUG)


class PlantVillageDatamodule(pl.LightningDataModule):
    def __init__(self, cache_path: str, data_path: str, train_batch_size: int, val_batch_size: int):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_path)
        self.data_dir = os.path.abspath(data_path)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        logging.info("Dataset folder was not found... Downloading data")
        dataset = load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=self.cache_dir)
        dataset.save_to_disk(self.data_dir)
        logging.info("Data downloaded successfully!")

    def setup(self, stage: str):
        data_dict = load_from_disk(self.data_dir)
        train_data = data_dict["train"]
        val_data = data_dict["test"]

        self.train_data = train_data.set_transform(train_transforms)
        self.val_data = val_data.set_transform(val_transforms)

    def train_dataloader(self):
        train_dataset = DataLoader(dataset=self.train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=collate) # pyright: ignore[reportArgumentType]
        return train_dataset

    def val_dataloader(self):
        val_dataset = DataLoader(dataset=self.val_data, batch_size=self.val_batch_size, shuffle=False, collate_fn=collate) # pyright: ignore[reportArgumentType]
        return val_dataset
