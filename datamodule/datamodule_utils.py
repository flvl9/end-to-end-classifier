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

def train_transforms(dataset: Dict[str, List[Any]])-> Dict[str, List[Any]]:
    """
    Applies on-the-fly augmentations to the training data.
    These augmentations simulate brightness variations, out of focus issues,
    and texture changes.
    args:
        dataset - A dictionary containing a batch of a HuggingFace dataset.
    returns:
        dataset - A new dictionary containing an "image_tensor" with the 
        transformations
    """
    

    dataset["image_tensor"] = [TRAIN_TRANSFORMATIONS(image) for image in dataset["image"]]
    
    return dataset

def val_transforms(dataset: Dict[str, List[Any]])-> Dict[str, List[Any]]:
    """
    Preprocesses the images by converting them to tensors and normalizing 
    the pixel values.
    args:
        dataset - A dictionary containing a batch of a HuggingFace dataset.
    returns:
        dataset: The same dictionary with a new "image_tensor" key-value with
        the normalized pixel values.
    """
    

    dataset["image_tensor"] = [VAL_TRANSFORMATIONS(image) for image in dataset["image"]]

    return dataset

def collate(dataset_batch: List[Dict[str, Any]])-> Dict[str, torch.Tensor]:
    tensors = [data["image_tensor"] for data in dataset_batch]
    labels = [data["label"] for data in dataset_batch]

    return {
        "image_tensor": torch.stack(tensors),
        "labels": torch.tensor(labels)
    }
