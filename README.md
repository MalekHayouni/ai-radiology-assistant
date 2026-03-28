# AI Radiology Assistant

> Multi-label pathology detection from chest X-rays using CheXNet DenseNet-121 pretrained on NIH Chest X-ray14.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Stack](https://img.shields.io/badge/stack-React%20%7C%20FastAPI%20%7C%20PyTorch%20%7C%20Docker-blue)
![Dataset](https://img.shields.io/badge/dataset-NIH%20ChestX--ray14-green)

---

## Problem

Radiologists face an overwhelming volume of chest X-rays daily, leading to diagnostic fatigue and delayed reporting especially in under-resourced hospitals. Manual interpretation is time-consuming and inconsistent across practitioners.

This project explores how a pretrained deep learning model (CheXNet, DenseNet-121) can assist by flagging suspected pathologies automatically, surfacing visual evidence via Grad-CAM heatmaps and presenting confidence scores  giving the radiologist a second opinion at inference speed.

---

## Architecture

The system is split into three layers:

**Frontend (React + Vite):** The radiologist uploads a chest X-ray image. The UI sends it to the backend via a REST call, then renders the returned heatmap overlay and per-pathology confidence scores side by side with the original image.

**Backend (FastAPI + PyTorch):** Receives the image, runs it through a preprocessing pipeline (resize to 224×224, ImageNet normalization), feeds it to the DenseNet-121 model, generates Grad-CAM activation maps via OpenCV and returns predictions as JSON.

**Model (DenseNet-121 / CheXNet):** Pretrained on NIH Chest X-ray14 (112,000 labeled images). The final classification layer is replaced with a 14-unit sigmoid head for multi-label output, one score per pathology class.

All services are containerized and orchestrated with Docker Compose.

```
┌────────────────────────────────────────────────────────────────┐
│                         Docker Compose                         │
│                                                                │
│  ┌──────────────────┐   REST    ┌──────────────────────────┐   │
│  │  React Frontend  │ ────────► │     FastAPI Backend      │   │
│  │  (Vite + Tailwind│ ◄──────── │                          │   │
│  │                  │   JSON    │  preprocess → DenseNet   │   │
│  │  - Upload UI     │           │  → Grad-CAM → response   │   │
│  │  - Heatmap viewer│           │                          │   │
│  │  - Scores panel  │           └──────────┬───────────────┘   │
│  └──────────────────┘                      │ inference         │
│                                            ▼                   │
│                               ┌────────────────────────┐       │
│                               │   DenseNet-121 Model   │       │
│                               │   (CheXNet weights)    │       │
│                               │   14-class sigmoid out │       │
│                               └────────────────────────┘       │
└────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | React (Vite), TailwindCSS | Upload UI, results rendering, heatmap viewer |
| Backend | FastAPI, Python 3.11 | REST API, request handling, response formatting |
| ML / Vision | PyTorch, torchvision, OpenCV | Model inference, Grad-CAM heatmap generation |
| Model | DenseNet-121 (CheXNet weights) | Multi-label pathology classification |
| Dataset | NIH Chest X-ray14 | 112,120 frontal-view X-rays, 14 disease labels |
| Deployment | Docker, Docker Compose | Containerized full-stack deployment |

---

## Expected Features

- Upload a chest X-ray (PNG/JPEG) through a clean web interface
- Receive multi-label predictions across 14 pathology classes with confidence scores
- View Grad-CAM heatmap overlaid on the original image, highlighting suspected regions
- See a ranked list of detected conditions sorted by confidence
- Health check endpoint for service monitoring
- Fully containerized — runs with a single `docker compose up`

---

## Project Structure

```
ai-radiology-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── model.py         # DenseNet-121 inference
│   │   ├── gradcam.py       # Grad-CAM heatmap generation
│   │   └── preprocess.py    # Image preprocessing pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadZone.jsx
│   │   │   ├── HeatmapViewer.jsx
│   │   │   └── ScoresPanel.jsx
│   │   └── App.jsx
│   ├── package.json
│   └── Dockerfile
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_model_research.ipynb
├── docker-compose.yml
└── README.md
```

---

## Pathology Classes (14)

`Atelectasis` · `Cardiomegaly` · `Effusion` · `Infiltration` · `Mass` · `Nodule` · `Pneumonia` · `Pneumothorax` · `Consolidation` · `Edema` · `Emphysema` · `Fibrosis` · `Pleural Thickening` · `Hernia`

---

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/ai-radiology-assistant
cd ai-radiology-assistant
docker compose up --build
```

Frontend → http://localhost:5173  
Backend API → http://localhost:8000  
API docs → http://localhost:8000/docs

---

## Roadmap

| Phase | Milestone | Status |
|---|---|---|
| 1 | Dataset exploration, project scaffolding, research notes | ✅ Done |
| 2 | DenseNet-121 model loader, preprocessing pipeline | 🔄 In Progress |
| 3 | FastAPI endpoints, Grad-CAM generation | 📋 Planned |
| 4 | React upload UI, heatmap viewer, scores panel | 📋 Planned |
| 5 | Model evaluation (AUC per class), Docker Compose, demo | 📋 Planned |

---

## References

- [CheXNet: Radiologist-Level Pneumonia Detection](https://arxiv.org/abs/1711.05225) 
- [NIH Chest X-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [torchvision DenseNet](https://pytorch.org/vision/stable/models.html)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

---

*Portfolio project — Malek Hayouni · ESPRIT School of Engineering · 2026*
