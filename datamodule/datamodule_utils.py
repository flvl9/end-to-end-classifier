import torch
from typing import List, Dict, Any
from torchvision import transforms

TRAIN_TRANSFORMATIONS = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(size=256),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
VAL_TRANSFORMATIONS = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def collate(dataset_batch: List[Dict[str, Any]])-> Dict[str, torch.Tensor]:
    """
    Defines how to combine the samples of the dataset into a mini-batch.
    args:
        dataset_batch: A list of examples from the dataset.
    returns:
        A dictionary containing the processed batch to feed the model.
    """
    tensors = [data["image_tensor"] for data in dataset_batch]
    labels = [data["label"] for data in dataset_batch]

    return {
        "image_tensor": torch.stack(tensors),
        "labels": torch.tensor(labels)
    }
