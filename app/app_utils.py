import os
import torch
import mlflow
import random
from PIL import Image
import streamlit as st
from typing import Tuple, List
from dotenv import load_dotenv
from torchvision import transforms
from mlflow.client import MlflowClient

load_dotenv()

MLF_TRACKING_URI = os.getenv("MLF_TRACKING_URI")
mlflow.set_tracking_uri(MLF_TRACKING_URI)

def preprocess_image(image) -> torch.Tensor:
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image).convert("RGB")
    img_tensor = transformations(img)

    return img_tensor.unsqueeze(0)

def retrieve_models_info() -> dict:
    client = MlflowClient(tracking_uri=MLF_TRACKING_URI)
    registered_models = client.search_registered_models()

    names, versions, traffics, scores = [], [], [], []

    for model in registered_models:
        if not model.latest_versions:
            continue

        ver = model.latest_versions[0]
        run = client.get_run(ver.run_id)
        mv = client.get_model_version(model.name, ver.version)

        names.append(model.name)
        versions.append(mv.version)
        traffics.append(float(mv.tags["traffic"]))
        scores.append(run.data.metrics.get("f1_Score", 0.0))

    return {
        "names": names,
        "versions": versions,
        "traffics": traffics,
        "scores": scores
    }

def normalize_traffic(traffics: list[float]) -> list[float]:
    normalized_traffics = []

    if sum(traffics) < 1:
        for traffic in traffics:
            normalized_traffics.append(traffic / sum(traffics))
        return normalized_traffics
    else:
        return traffics
    
@st.cache_resource
def load_models(names: list, versions: list) -> dict:
    cached_models = {}

    for name, version in zip(names, versions):
        uri = f"models:/{name}/{version}"
        cached_models[name] = mlflow.pytorch.load_model(uri)

    return cached_models

@st.cache_resource(scope="session")
def select_model(models_data: dict) -> dict:
    cache = load_models(models_data["names"], models_data["versions"])
    selected_model_name = random.choices(
        models_data["names"],
        weights=models_data["traffics"],
        k=1)[0]
    model_idx = models_data["names"].index(selected_model_name)

    return {
        "model": cache[selected_model_name],
        "name": selected_model_name,
        "version": models_data["versions"][model_idx],
        "traffic": models_data["traffics"][model_idx],
        "score": models_data["scores"][model_idx]
    }

def make_prediction(model: torch.nn.Module, img_tensor: torch.Tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        clas = torch.argmax(probs, dim=1).item()
        classes = int(clas)
        score = probs[0, classes]

    return classes, score.item()

def format_label(label: str) -> Tuple[str, str]:
    raw_crop, raw_disease = label.split("__")
    if "_" in raw_crop:
        crop = f"{raw_crop.split("_")[0]} {raw_crop.split("_")[1]}"
    else:
        crop = raw_crop

    disease_chunks = raw_disease.split("_")

    disease = ""
    for i in range(len(disease_chunks)):
        disease += " " + disease_chunks[i]

    return crop, disease

def format_treatment(treatment: str) -> List[str]:
    treatment_list = [chunk for chunk in treatment.split(";")]

    return treatment_list
