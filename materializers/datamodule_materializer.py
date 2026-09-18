import os
import json
from typing import Any
from zenml.enums import ArtifactType
from datamodule.datamodule import PlantVillageDatamodule
from zenml.materializers.base_materializer import BaseMaterializer

class PlantVillageMaterializer(BaseMaterializer):
    """
    A custom materializer for the PlantVillageDatamodule.
    """
    ASSOCIATED_TYPES = (PlantVillageDatamodule,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save(self, data:PlantVillageDatamodule) -> None:
        os.makedirs(self.uri, exist_ok=True)
        config = {
            "cache_dir": data.cache_dir,
            "data_dir": data.data_dir,
            "train_batch_size": data.train_batch_size,
            "val_batch_size": data.val_batch_size,
            "n_workers": data.n_workers
        }
        filepath = os.path.join(self.uri, "datamodule.json")
        with open(filepath, "w") as file:
            json.dump(config, file)

    def extract_metadata(self, data: PlantVillageDatamodule) -> dict[str, Any]:
        return {
            "dataset_name": "GVJahnavi/PlantVillage_dataset",
            "num_classes": data.num_classes,
        }

    def load(self, data_type: type) -> PlantVillageDatamodule:
        filepath = os.path.join(self.uri, "datamodule.json")
        with open(filepath, "r") as file:
            config = json.load(file)
        return PlantVillageDatamodule(**config)
