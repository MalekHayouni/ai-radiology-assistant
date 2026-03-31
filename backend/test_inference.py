import torch
from PIL import Image
from app.model import RadiologyModel
from app.preprocess import preprocess_image

model = RadiologyModel()


image = Image.open("test_xray.jpg").convert("RGB")
tensor = preprocess_image(image)

predictions = model.predict(tensor)

print("\nTop 5 predictions:")
for p in predictions[:5]:
    print(f"  {p['class']:<22} {p['confidence']:.4f}")