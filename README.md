# GeoGuessr Concept Bottleneck Model for Geolocation

A research framework for interpretable geolocation prediction from Google Street View images using Concept Bottleneck Models (CBMs). The system learns geographic concepts through a hierarchical, interpretable architecture built on GeoGuessr metadata.

## Overview

Unlike black-box approaches, this framework enforces a concept bottleneck where predictions flow through learned concept representations, enabling both accuracy and interpretability. The three-stage training pipeline progresses from domain pretraining → concept learning → geolocation prediction.

**Key Features**: Text-anchored prototypes, hierarchical concept structure, cross-attention interpretability, semantic geocells, and automated GeoGuessr bot.

## Architecture

**Three-Stage Pipeline**:

1. **Stage 0 - Domain Pretraining**: Partially unfreezes StreetCLIP vision encoder, aligning image features with GPS coordinates and concept domains
2. **Stage 1 - Concept Learning**: Trains concept bottleneck (MLP/Transformer) with text-anchored prototypes to predict hierarchical concepts
3. **Stage 2 - Geolocation**: Cross-attention mechanism where concept embeddings query image patches to predict location via geocell classification

**Core Components**: StreetCLIP encoder → Concept bottleneck → Text prototypes → Cross-attention module → Geocell system

## Quick Start

### Installation
```bash
conda env create -f environment.yml
conda activate geoguessr-cbm
pip install -r requirements.txt
```

Add your LearnableMeta API key to `misc/constants.py`.

### Data Pipeline
```bash
# Scrape metadata, download panoramas, preprocess
python scripts/data_collection/scrape_lm.py
python scripts/data_collection/download_pano.py
python scripts/data_processing/preprocess_panorama.py --panorama-folder data/{geoguessrId}/panorama
python scripts/data_processing/enrich_metas_with_coords.py
```

### Training
```bash
# Stage 0: Domain pretraining
python scripts/training/train_stage0_prototype.py --csv_path data/dataset.csv --stage0_epochs 20

# Stage 1: Concept learning
python scripts/training/train_stage1_prototype.py --csv_path data/dataset.csv \
    --resume_from_checkpoint results/stage0-.../best_model_stage0.pt

# Stage 2: Geolocation
python scripts/training/train_stage2_cross_attention.py --csv_path data/dataset.csv \
    --stage1_checkpoint results/stage1-.../best_model_stage1.pt
```

### Evaluation & Inference
```bash
# Evaluate models
python scripts/evaluation/eval_stage1_on_split.py --checkpoint results/stage1-.../best_model_stage1.pt
python scripts/evaluation/eval_stage2_on_split.py --checkpoint results/stage2-.../best_model_stage2_xattn.pt

# Single image prediction
python scripts/inference/predict_location.py --image_path path/to/image.jpg \
    --checkpoint results/stage2-.../best_model_stage2_xattn.pt
```

## Repository Structure

```
├── scripts/
│   ├── data_collection/        # Scraping and downloading
│   ├── data_processing/        # Preprocessing and enrichment
│   ├── training/               # Stage 0/1/2 training scripts
│   ├── evaluation/             # Evaluation and benchmarking
│   ├── analysis/               # Visualization and interpretability
│   └── inference/              # Single image prediction
├── src/
│   ├── models/                 # Model architectures
│   ├── dataset.py              # PanoramaCBMDataset
│   ├── losses.py               # Loss functions
│   └── concepts/               # Concept hierarchy utilities
├── geoguessr_bot/              # Automated gameplay bot
├── docs/                       # Detailed documentation
└── jobs/                       # SLURM job scripts
```

## Dataset & Model

**Dataset**: Each sample contains a 336×336 panorama image, hierarchical concept labels (meta + parent), country, coordinates, and metadata. Automatically handles train/val/test splits with concept stratification.

**Concept Bottleneck**: Enforces strict separation—Stage 1 uses only concept embeddings, Stage 2 uses embeddings as cross-attention queries. Hierarchical structure includes ~100+ fine-grained meta concepts grouped into ~10-20 parent categories.

**Text-Anchored Prototypes**: Initialized from CLIP text encodings with learnable residuals, using cosine similarity classification with per-concept calibration.

## Evaluation Metrics

**Concept Classification**: Top-1/Top-5 accuracy, macro-averaged recall, semantic-close accuracy

**Geolocation**: Median distance error, threshold accuracies (street: 1km, city: 25km, region: 200km, country: 750km, continent: 2500km), geocell accuracy

## GeoGuessr Bot

Complete automated gameplay system with Chrome extension, Flask API server, and GUI controller. See `geoguessr_bot/README.md` for setup.

## Documentation

- [Architecture](docs/README_ARCHITECTURE.md) - Detailed pipeline and design decisions
- [CBM System](docs/README_cbm.md) - Complete system guide and data flow
- [Dataset](docs/README_dataset.md) - Dataset structure and usage

## Requirements

- Python 3.8+, PyTorch 2.0+, CUDA-capable GPU
- LearnableMeta API access for data collection
