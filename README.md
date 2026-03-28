#  AI Radiology Assistant
> Multi-label pathology detection from chest X-rays using CheXNet DenseNet-121 pretrained on NIH Chest X-ray14.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Stack](https://img.shields.io/badge/stack-React%20%7C%20FastAPI%20%7C%20PyTorch%20%7C%20Docker-blue)
![Dataset](https://img.shields.io/badge/dataset-NIH%20ChestX--ray14-green)

## Overview
This portfolio project demonstrates an end-to-end AI-assisted radiology pipeline.
A radiologist uploads a chest X-ray; the system returns:
- Multi-label pathology predictions (14 classes)
- Grad-CAM heatmap overlays highlighting suspected regions
- Confidence scores per condition

## Tech Stack
| Layer      | Technology                          |
|------------|-------------------------------------|
| Frontend   | React (Vite), TailwindCSS           |
| Backend    | FastAPI, PyTorch, OpenCV            |
| Model      | DenseNet-121 (CheXNet weights)      |
| Deployment | Docker, Docker Compose              |
| Dataset    | NIH Chest X-ray14 (112,000 images)  |

## Features
- [x] Dataset exploration & statistics
- [x] Project scaffolding
- [ ] DenseNet-121 inference pipeline
- [ ] Grad-CAM heatmap generation
- [ ] FastAPI REST endpoints
- [ ] React upload + results UI
- [ ] Docker Compose deployment
- [ ] Model evaluation (AUC per class)

## Getting Started
```bash
git clone https://github.com/YOUR_USERNAME/ai-radiology-assistant
cd ai-radiology-assistant
docker compose up --build
```

## Project Structure
```
ai-radiology-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── model.py         # DenseNet-121 inference
│   │   └── gradcam.py       # Grad-CAM heatmaps
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── App.jsx
│   └── Dockerfile
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_model_research.ipynb
├── docker-compose.yml
└── README.md
```

## Pathology Classes
Atelectasis · Cardiomegaly · Effusion · Infiltration · Mass ·
Nodule · Pneumonia · Pneumothorax · Consolidation · Edema ·
Emphysema · Fibrosis · Pleural Thickening · Hernia

## References
- [CheXNet Paper](https://arxiv.org/abs/1711.05225) — Rajpurkar et al., 2017
- [NIH Chest X-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [torchvision DenseNet](https://pytorch.org/vision/stable/models.html)
