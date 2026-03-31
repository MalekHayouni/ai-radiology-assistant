from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image

from app.model import RadiologyModel
from app.preprocess import preprocess_image
from app.gradcam import generate_gradcam

app = FastAPI(
    title="AI Radiology Assistant",
    description="Multi-label chest X-ray pathology detection using CheXNet DenseNet-121",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once at startup
model = RadiologyModel()


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "DenseNet-121 (CheXNet)", "version": "0.1.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # preprocess
    tensor = preprocess_image(image)

    # inference
    predictions = model.predict(tensor)

    # gradcam
    heatmap_b64 = generate_gradcam(model, tensor, image)

    return JSONResponse({
        "predictions": predictions,
        "heatmap": heatmap_b64,
    })