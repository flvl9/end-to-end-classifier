import torch
import logging
import numpy as np
import lightning as pl
from datasets import DatasetDict
from torchvision import transforms
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
from .datamodule_utils import TRAIN_TRANSFORMATIONS, VAL_TRANSFORMATIONS, collate

logging.basicConfig(level=logging.DEBUG)

class PlantVillageDataset(Dataset):
    """
    A custom dataset for each split of the DatasetDict.
    It enables on-the-fly augmentation, and the required
    transformations for the images.
    args:
        hf_dataset - The corresponding HuggingFace dataset.
        transform - The transformations to be applied to the images.
    """
    def __init__(
            self,
            hf_dataset: HFDataset,
            transform: transforms.Compose
            ):
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
    Creates a custom Datamodule for the lightning Trainer.
    args:
        dataset - The dataset containing the train/val splits.
        train_batch_size - Batch size for the training dataloader.
        val_batch_size - Batch size for the validation dataloader.
        n_workers: Number of subprocesses to load data in parallel.
    """
    def __init__(
            self,
            dataset: DatasetDict,
            train_batch_size: int,
            val_batch_size: int,
            n_workers: int
            ):
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.n_workers = n_workers
        self._class_weights = None

    def setup(self, stage: str):
        # Only "fit" stage is used
        self.train_dataset = PlantVillageDataset(self.dataset["train"], TRAIN_TRANSFORMATIONS)
        self.val_dataset = PlantVillageDataset(self.dataset["test"], VAL_TRANSFORMATIONS)

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

    def train_dataloader(self) -> DataLoader:
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

    def val_dataloader(self) -> DataLoader:
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
