import os
import lightning as pl
from dotenv import load_dotenv
from datasets import load_dataset
from torch.utils.data import DataLoader
from datamodule_utils import train_transforms, val_transforms, collate

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class PlantVillageDatamodule(pl.LightningDataModule):
    def __init__(self, data_path: str, train_batch_size: int, val_batch_size: int):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=self.data_dir)

    def setup(self, stage: str):
        self.train_data = load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=self.data_dir, split="train")
        self.val_data = load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=self.data_dir, split="test")

        self.train_data.set_transform(train_transforms)
        self.val_data.set_transform(val_transforms)

    def train_dataloader(self):
        train_dataset = DataLoader(dataset=self.train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=collate) # pyright: ignore[reportArgumentType]
        return train_dataset

    def val_dataloader(self):
        val_dataset = DataLoader(dataset=self.val_data, batch_size=self.val_batch_size, shuffle=False, collate_fn=collate) # pyright: ignore[reportArgumentType]
        return val_dataset
    