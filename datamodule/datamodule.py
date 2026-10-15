import os
from typing import Any
import torch
import logging
import numpy as np
import lightning as pl
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from .datamodule_utils import TRAIN_TRANSFORMATIONS, VAL_TRANSFORMATIONS, collate

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logging.basicConfig(level=logging.DEBUG)


class PlantVillageDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = self.transform(example["image"])
        label = example ["label"]
        return {
            "image_tensor": image,
            "label": label
        }


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

        self.train_dataset = PlantVillageDataset(data_dict["train"], TRAIN_TRANSFORMATIONS)
        self.val_dataset = PlantVillageDataset(data_dict["test"], VAL_TRANSFORMATIONS)

    @property
    def class_weights(self) -> torch.Tensor:
        if self._class_weights is None:
            labels = np.array(self.train_dataset["label"])
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
            dataset=self.train_dataset,
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
            dataset=self.val_dataset,
            batch_size=self.val_batch_size, 
            shuffle=False, 
            collate_fn=collate,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True
            )
        return val_dataset
