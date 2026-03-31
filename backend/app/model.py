import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict

PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]

WEIGHTS_PATH = "weights/chexnet.pth"


class CheXNet(nn.Module):
    
    #DenseNet-121 with a 14-class sigmoid output head replicating the CheXNet architecture 
    

    def __init__(self, num_classes: int = 14):
        super(CheXNet, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # replace classifier with 14 unit sigmoid head
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


class RadiologyModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CheXNet(num_classes=14).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
            fixed = {"densenet." + k: v for k, v in state_dict.items()}
            self.model.load_state_dict(fixed, strict=False)
            print("[model] Weights loaded successfully from", WEIGHTS_PATH)
        except FileNotFoundError:
            print(f"[model] Warning: {WEIGHTS_PATH} not found. Using ImageNet pretrained weights.")

    def predict(self, tensor: torch.Tensor) -> List[Dict]:
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.model(tensor)  # shape: (1, 14)

        scores = output.squeeze(0).cpu().tolist()
        predictions = [
            {"class": cls, "confidence": round(score, 4)}
            for cls, score in zip(PATHOLOGY_CLASSES, scores)
        ]
        # sort by confidence descending
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions