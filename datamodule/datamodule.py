import os
import torch
import logging
import numpy as np
import lightning as pl
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from .datamodule_utils import train_transforms, val_transforms, collate

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logging.basicConfig(level=logging.DEBUG)


class PlantVillageDatamodule(pl.LightningDataModule):
    """
    Defines the Data Module required by the lightning trainer.
    Contains the logic of how to download and preprocess the dataset, returning
    the dataloaders for the training and validation stages.
    args:
        cache_dir - The directory where the dataset cache will be stored.
        data_dir - The directory where the dataset will be created and stored.
        train_batch_size - Batch size for the training dataloader.
        val_batch_size - Batch size for the validation dataloader.
        n_workers: Number of subprocesses to load data in parallel.
    """
    def __init__(
            self,
            cache_dir: str,
            data_dir: str,
            train_batch_size: int,
            val_batch_size: int,
            n_workers: int
            ):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.n_workers = n_workers
        self._class_weights = None

    def prepare_data(self):
        if os.path.exists(self.data_dir):
            logging.info("Dataset found on disk!")
        else:
            logging.info("Dataset folder was not found... Downloading data")
            dataset = load_dataset("GVJahnavi/PlantVillage_dataset", cache_dir=self.cache_dir)
            dataset.save_to_disk(self.data_dir)
            logging.info("Data downloaded successfully!")

    def setup(self, stage: str):
        # Only "fit" stage is used
        data_dict = load_from_disk(self.data_dir)
        self.train_data = data_dict["train"]
        # In this case, the "test" split serves as the validation data
        self.val_data = data_dict["test"]

        self.train_data.set_transform(train_transforms)
        self.val_data.set_transform(val_transforms)

    @property
    def class_weights(self) -> torch.Tensor:
        if self._class_weights is None:
            labels = np.array(self.train_data["label"])
            self.num_classes = len(np.unique(labels))
            counts = np.bincount(labels, minlength=self.num_classes)
            counts[counts == 0] = 1 # Avoids division by zero in case of absent classes
            self._class_weights = torch.tensor(
                len(labels) / (self.num_classes * counts),
                dtype=torch.float
                )
        return self._class_weights

    def train_dataloader(self):
        train_dataset = DataLoader(
            dataset=self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True
            )
        return train_dataset

    def val_dataloader(self):
        val_dataset = DataLoader(
            dataset=self.val_data,
            batch_size=self.val_batch_size, 
            shuffle=False, 
            collate_fn=collate,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True
            )
        return val_dataset
