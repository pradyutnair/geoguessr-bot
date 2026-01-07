# GeoGuessr Concept Bottleneck Model for Geolocation

This repository implements a comprehensive research framework for interpretable geolocation prediction using Concept Bottleneck Models (CBMs) trained on Google Street View panorama images. The system leverages GeoGuessr meta information to build machine learning models that learn geographic concepts and their relationships to spatial coordinates through a hierarchical, interpretable architecture.

## Overview

The project addresses the challenge of geolocation prediction from street view imagery by learning interpretable visual concepts that correlate with geographic locations. Unlike black-box approaches, this framework enforces a concept bottleneck architecture where predictions are made through learned concept representations, enabling both high accuracy and interpretability.

The system features a three-stage training pipeline that progresses from domain pretraining through concept learning to geolocation prediction. Each stage builds upon the previous, with frozen components ensuring that downstream tasks operate exclusively on concept embeddings rather than raw image features.

## Key Features

- **Three-Stage Training Pipeline**: Sequential training from domain alignment to concept learning to geolocation prediction
- **Hierarchical Concept Structure**: Two-level concept hierarchy with fine-grained meta concepts and coarse parent categories
- **Text-Anchored Prototypes**: Concept representations initialized from CLIP's multimodal space with learnable residuals
- **Cross-Attention Interpretability**: Patch-level attention visualization showing which image regions contribute to predictions
- **Semantic Geocells**: Per-country K-Means clustering in 3D space for structured geolocation prediction
- **Complete Data Pipeline**: Tools for scraping, processing, and enriching GeoGuessr metadata
- **Automated Bot System**: Chrome extension and API server for real-time GeoGuessr gameplay
- **Comprehensive Evaluation**: Multiple metrics for concept classification and geolocation accuracy

## Architecture

### Three-Stage Training Pipeline

**Stage 0: Domain Contrastive Pretraining**
- Partially unfreezes top layers of StreetCLIP vision encoder
- Aligns image features with GPS coordinates and concept domains
- Trains image-GPS contrastive, image-concept contrastive, and hierarchical consistency losses
- Establishes foundation for concept learning

**Stage 1: Text-Prototype Concept Learning**
- Freezes image encoder from Stage 0
- Trains concept bottleneck (MLP or Transformer) to predict hierarchical concepts
- Uses text-anchored prototypes with learnable residuals for semantic grounding
- Classification via cosine similarity to prototypes with per-concept calibration

**Stage 2: Cross-Attention Geolocation**
- Freezes image encoder and concept bottleneck from previous stages
- Trains cross-attention mechanism where concept embeddings query image patches
- Predicts geolocation through semantic geocell classification and offset regression
- Provides interpretable attention maps showing spatial concept activation

### Model Components

- **Image Encoder**: StreetCLIP vision transformer for extracting visual features
- **Concept Bottleneck**: MLP or Transformer mapping image features to concept embeddings
- **Text Prototypes**: CLIP-encoded concept descriptions with learnable residual adjustments
- **Cross-Attention Module**: Multi-head attention mechanism for concept-patch fusion
- **Geocell System**: Per-country K-Means clustering for structured location prediction

## Repository Structure

```
├── scripts/
│   ├── data_collection/          # Data scraping and downloading
│   │   ├── scrape_lm.py          # Scrape meta information from learnablemeta.com
│   │   ├── scrape_lm_async.py    # Async version for faster scraping
│   │   ├── download_pano.py      # Download panorama images
│   │   └── download_pano_parallel.py  # Parallel download implementation
│   ├── data_processing/          # Data preprocessing and enrichment
│   │   ├── preprocess_panorama.py  # Crop and resize panorama images
│   │   └── enrich_metas_with_coords.py  # Add coordinate information to metadata
│   ├── training/                 # Training scripts for each stage
│   │   ├── train_stage0_prototype.py  # Stage 0 pretraining
│   │   ├── train_stage1_prototype.py  # Stage 1 concept learning
│   │   ├── train_stage2_cross_attention.py  # Stage 2 geolocation
│   │   └── train_concept_aware.py  # Alternative training approach
│   ├── evaluation/               # Evaluation and benchmarking
│   │   ├── eval_stage1_on_split.py  # Evaluate concept classification
│   │   ├── eval_stage2_on_split.py  # Evaluate geolocation accuracy
│   │   └── baseline_geoclip_streetclip.py  # Baseline model comparisons
│   ├── analysis/                 # Analysis and visualization tools
│   │   ├── visualize_geocells_map.py  # Visualize semantic geocells
│   │   ├── interpretability_plots.py  # Generate attention visualizations
│   │   └── compare_vanilla_vs_finetuned_latest.py  # Model comparison
│   └── inference/                # Inference and deployment
│       └── predict_location.py   # Single image geolocation prediction
├── src/                          # Core implementation
│   ├── models/                   # Model architectures
│   │   ├── concept_aware_cbm.py  # Three-stage CBM implementation
│   │   ├── streetclip_encoder.py  # StreetCLIP encoder wrapper
│   │   ├── baseline.py           # Baseline model implementations
│   │   └── encoder_factory.py    # Encoder initialization utilities
│   ├── dataset.py                # PanoramaCBMDataset implementation
│   ├── losses.py                 # Loss function definitions
│   ├── evaluation.py             # Evaluation metrics and utilities
│   ├── config.py                 # Configuration dataclasses
│   └── concepts/                 # Concept extraction utilities
│       └── utils.py             # Concept hierarchy building
├── geoguessr_bot/               # Automated GeoGuessr bot
│   ├── chrome_extension/         # Chrome extension for gameplay
│   ├── api_server.py             # Flask API server for model inference
│   ├── gui_controller.py        # GUI automation for game interaction
│   └── geoguessr_api.py         # GeoGuessr API client
├── docs/                         # Comprehensive documentation
│   ├── README_ARCHITECTURE.md    # Detailed architecture documentation
│   ├── README_cbm.md             # CBM system guide
│   ├── README_dataset.md         # Dataset documentation
│   └── README_project.md         # Original project setup guide
├── jobs/                         # SLURM job scripts for cluster execution
│   ├── stage0/                   # Stage 0 training jobs
│   ├── stage1/                   # Stage 1 training jobs
│   ├── stage2/                   # Stage 2 training jobs
│   ├── evaluation/               # Evaluation job scripts
│   └── bot/                      # Bot deployment jobs
├── data/                         # Data directory (created by scripts)
├── results/                      # Training outputs and checkpoints
├── environment.yml               # Conda environment specification
├── requirements.txt              # Python package requirements
└── README.md                     # This file
```

## Quick Start

### Installation

1. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate geoguessr-cbm
   ```

2. **Install additional requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys:**
   Add your LearnableMeta API key to `misc/constants.py` for data collection.

### Data Collection

1. **Scrape meta information:**
   ```bash
   python scripts/data_collection/scrape_lm.py
   ```

2. **Download panorama images:**
   ```bash
   python scripts/data_collection/download_pano.py
   ```

3. **Preprocess images:**
   ```bash
   python scripts/data_processing/preprocess_panorama.py \
       --panorama-folder data/{geoguessrId}/panorama
   ```

4. **Enrich metadata with coordinates:**
   ```bash
   python scripts/data_processing/enrich_metas_with_coords.py
   ```

### Training

1. **Stage 0: Domain Pretraining**
   ```bash
   python scripts/training/train_stage0_prototype.py \
       --csv_path data/dataset.csv \
       --stage0_epochs 20 \
       --unfreeze_layers 2
   ```

2. **Stage 1: Concept Learning**
   ```bash
   python scripts/training/train_stage1_prototype.py \
       --csv_path data/dataset.csv \
       --resume_from_checkpoint results/stage0-.../best_model_stage0.pt \
       --stage1_epochs 50
   ```

3. **Stage 2: Geolocation Prediction**
   ```bash
   python scripts/training/train_stage2_cross_attention.py \
       --csv_path data/dataset.csv \
       --stage1_checkpoint results/stage1-.../best_model_stage1.pt \
       --epochs 30 \
       --ablation_mode both
   ```

### Evaluation

```bash
# Evaluate concept classification (Stage 1)
python scripts/evaluation/eval_stage1_on_split.py \
    --checkpoint results/stage1-.../best_model_stage1.pt

# Evaluate geolocation accuracy (Stage 2)
python scripts/evaluation/eval_stage2_on_split.py \
    --checkpoint results/stage2-.../best_model_stage2_xattn.pt
```

### Inference

```bash
# Predict location for a single image
python scripts/inference/predict_location.py \
    --image_path path/to/image.jpg \
    --checkpoint results/stage2-.../best_model_stage2_xattn.pt
```

## Dataset

The `PanoramaCBMDataset` loads processed GeoGuessr metadata and panorama images. Each sample contains:

- **Image**: Preprocessed 336x336 panorama image
- **Meta Concept**: Fine-grained concept label (e.g., "ACT gap reflector")
- **Parent Concept**: Coarse category (e.g., "Urban", "Rural")
- **Country**: Country name
- **Coordinates**: Latitude and longitude in degrees
- **Metadata**: Additional notes and information

The dataset automatically handles train/val/test splits with stratification by concept to ensure all concepts appear in training.

## Model Architecture Details

### Concept Bottleneck Design

The architecture enforces strict concept bottleneck principles:
- Stage 1 predictions use only concept embeddings, not raw image features
- Stage 2 geolocation uses concept embeddings as queries in cross-attention
- All downstream tasks operate on the 512-dimensional concept bottleneck

### Hierarchical Concept Learning

- **Meta Concepts**: Fine-grained labels extracted from GeoGuessr meta names (~100+ concepts)
- **Parent Concepts**: Coarse categories grouping related meta concepts (~10-20 concepts)
- **Hierarchical Supervision**: Multiple losses ensure consistency between levels

### Text-Anchored Prototypes

- Prototypes initialized from CLIP text encodings of concept names
- Learnable residuals allow adaptation to visual patterns
- Per-concept bias and logit scale provide calibration
- Cosine similarity classification maintains semantic interpretability

### Cross-Attention Mechanism

- Concept embeddings serve as queries
- Image patch tokens serve as keys and values
- Attention weights provide spatial interpretability
- Fusion mechanisms balance concept and spatial information

## Evaluation Metrics

### Concept Classification (Stage 1)
- Top-1 and Top-5 accuracy for meta and parent concepts
- Macro-averaged recall
- Semantic-close accuracy (within similarity threshold)

### Geolocation (Stage 2)
- Median distance error (kilometers)
- Threshold accuracies:
  - Street: 1 km
  - City: 25 km
  - Region: 200 km
  - Country: 750 km
  - Continent: 2500 km
- Geocell classification accuracy

## Automated GeoGuessr Bot

The repository includes a complete bot system for automated GeoGuessr gameplay:

- **Chrome Extension**: Captures screenshots and submits predictions
- **API Server**: Flask server running Stage 2 model for inference
- **GUI Controller**: Alternative approach using GUI automation
- **SSH Tunnel Support**: Run model on remote GPU while playing locally

See `geoguessr_bot/README.md` for detailed setup instructions.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Architecture Documentation](docs/README_ARCHITECTURE.md)**: Detailed explanation of the three-stage pipeline, model components, and design decisions
- **[CBM System Guide](docs/README_cbm.md)**: Complete guide to the concept bottleneck model system, data flow, and coordinate handling
- **[Dataset Documentation](docs/README_dataset.md)**: Dataset structure, creation pipeline, and usage examples
- **[Project Setup](docs/README_project.md)**: Original project setup and data collection instructions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- Conda or pip for package management
- Access to LearnableMeta API for data collection

## License

See individual documentation files for licensing information.

## Citation

If you use this code in your research, please cite the associated paper or repository.
