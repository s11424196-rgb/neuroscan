# NeuroScan AI — Brain Tumour & Alzheimer's Detection System

> ⚠️ **Disclaimer**: This tool is for **research and educational use only** and is **not a substitute for professional medical diagnosis**. Any clinical deployment would require regulatory clearance and clinical validation.

## Overview

NeuroScan AI is an AI-assisted system that analyses brain MRI scans to:
1. **Detect and classify brain tumours** — glioma, meningioma, pituitary tumour, or no tumour
2. **Screen for Alzheimer's disease** — non-demented, very mild, mild, or moderate dementia

Users upload an MRI scan and receive a prediction, confidence score, and visual explanation (Grad-CAM heatmap) showing which regions of the scan influenced the result.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│   Streamlit  │────▶│   FastAPI    │────▶│   ML Pipeline     │
│   Frontend   │◀────│   Backend    │◀────│   (PyTorch)       │
└──────────────┘     └──────────────┘     └───────────────────┘
                           │                       │
                     ┌─────┴─────┐          ┌──────┴──────┐
                     │  SQLite   │          │   Model     │
                     │  (audit)  │          │  Registry   │
                     └───────────┘          └─────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip or uv

### Installation

```bash
# Clone and install
cd brain-mri-ai
pip install -e ".[dev]"

# Generate synthetic test data
python scripts/generate_synthetic_data.py

# Start the API server (mock mode — no trained models needed)
MOCK_INFERENCE=true uvicorn src.api.main:app --reload --port 8000

# In another terminal, start the Streamlit frontend
streamlit run src/frontend/app.py
```

### Demo Mode

The system includes a built-in demo mode that works without any trained models. Set `MOCK_INFERENCE=true` for the backend, or use the "Demo Mode" toggle in the Streamlit UI. This generates realistic-looking predictions with synthetic Grad-CAM heatmaps.

## Project Structure

```
brain-mri-ai/
├── configs/                    # Training and app configuration
│   └── training_config.yaml
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   ├── processed/              # Preprocessed data
│   └── synthetic/              # Generated test data
├── docs/                       # Documentation
├── models/
│   └── registry/               # Saved model checkpoints
├── reports/                    # Generated reports
├── scripts/
│   ├── generate_synthetic_data.py
│   └── train.py
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── core/               # Config, security
│   │   ├── models/             # Pydantic schemas
│   │   ├── routers/            # API endpoints
│   │   └── services/           # Business logic
│   ├── frontend/               # Streamlit app
│   │   ├── .streamlit/         # Streamlit config
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/              # App pages
│   │   └── utils/              # API client, mock inference
│   ├── ml/                     # Machine learning pipeline
│   │   ├── evaluation/         # Metrics, inference pipeline
│   │   ├── explainability/     # Grad-CAM
│   │   ├── models/             # Model architectures
│   │   ├── preprocessing/      # Image transforms, DICOM handling
│   │   └── training/           # Training loop, datasets
│   └── utils/                  # Shared utilities
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Training Models

```bash
# Train tumour classifier
python scripts/train.py \
    --task tumour \
    --backbone resnet50 \
    --data_dir data/synthetic/tumour \
    --output_dir models/registry/tumour_v1 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001

# Train Alzheimer's classifier
python scripts/train.py \
    --task alzheimer \
    --backbone efficientnet_b0 \
    --data_dir data/synthetic/alzheimer \
    --output_dir models/registry/alzheimer_v1 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload` | Upload a scan |
| POST | `/api/v1/upload/batch` | Upload multiple scans |
| GET | `/api/v1/scans/{scan_id}` | Get scan info |
| POST | `/api/v1/predict/{scan_id}` | Run prediction |
| GET | `/api/v1/results/{scan_id}` | Get prediction results |
| GET | `/api/v1/results/{scan_id}/heatmap` | Get heatmap image |
| GET | `/api/v1/reports/{scan_id}/pdf` | Download PDF report |
| GET | `/api/v1/reports/{scan_id}/csv` | Download CSV results |
| GET | `/api/v1/admin/model-info` | Model information |
| GET | `/api/v1/admin/audit-log` | Audit log |
| GET | `/health` | Health check |

## Datasets

| Task | Dataset | Notes |
|------|---------|-------|
| Tumour Classification | Kaggle Brain Tumor MRI | 4-class, 2D slices |
| Tumour Segmentation | BraTS Challenge | 3D multi-modal with segmentation masks |
| Alzheimer's | OASIS / Kaggle Alzheimer's MRI | 4-class staging |

⚠️ **Important**: All train/test splits are done by **patient ID**, never by image, to prevent data leakage.

## Model Performance Targets

| Task | Metric | Target |
|------|--------|--------|
| Tumour Classification | Macro F1 | ≥ 0.92 |
| Tumour Classification | Sensitivity/class | ≥ 0.90 |
| Alzheimer's Staging | Macro F1 | ≥ 0.80 |
| Alzheimer's (binary) | AUROC | ≥ 0.90 |
| Calibration | ECE | ≤ 0.05 |

## Ethics & Safety

- This is a **decision-support and educational tool**, not a diagnostic device
- All predictions include confidence scores and uncertainty estimates
- Low-confidence results are flagged for human review
- DICOM uploads are automatically anonymised
- Model limitations and failure cases are documented
- Subgroup performance is tracked for bias detection

## Technology Stack

- **ML**: PyTorch, torchvision, scikit-learn
- **Backend**: FastAPI, Pydantic, SQLite
- **Frontend**: Streamlit
- **Medical Imaging**: pydicom, nibabel, SimpleITK
- **Explainability**: Grad-CAM / Grad-CAM++
- **Deployment**: Docker, Docker Compose

## License

MIT — See [LICENSE](LICENSE) for details.

