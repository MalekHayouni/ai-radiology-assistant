import torch
import torchvision.transforms as transforms
from PIL import Image

# imagenet normalization stats 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    preprocess a PIL Image for DenseNet-121 inference

    steps:
    - Resize to 224x224 (CheXNet input size)
    - Convert to tensor [0, 1]
    - Normalize with ImageNet mean/std

    returns:
        torch.tensor of shape (1, 3, 224, 224)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = preprocess_pipeline(image)
    return tensor.unsqueeze(0)  