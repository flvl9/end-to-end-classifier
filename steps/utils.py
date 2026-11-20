import os
from dotenv import load_dotenv
from mlflow.client import MlflowClient

load_dotenv()

MLF_TRACKING_URI = os.getenv("MLF_TRACKING_URI")

def get_allocated_traffic() -> float:
    """
    Retrieves the total allocated traffic to the models in the 
    registry.
    returns:
        float - The amount of allocated traffic. 
    """
    traffic_sum = 0
    client = MlflowClient(tracking_uri=MLF_TRACKING_URI)
    registered_models = client.search_registered_models()

    for model in registered_models:
        if not model.latest_versions:
            continue

        ver = model.latest_versions[0]
        mv = client.get_model_version(model.name, ver.version)
        t = mv.tags.get("traffic")
        if t:
            traffic_sum += float(t)

    return traffic_sum
