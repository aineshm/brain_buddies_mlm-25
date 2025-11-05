# Candida Albicans ML Pipeline

**Deep Learning Pipeline for Cell Morphology Segmentation, Tracking, and Analysis**

UW Madison Machine Learning Marathon 2025 - Brain Buddies Team

---

## 🎯 Overview

This pipeline implements a complete deep learning solution for analyzing Candida albicans morphological transitions in time-lapse microscopy videos. It addresses all competition deliverables:

### Minimal Deliverables (6 points):
1. **Dispersal Initiation Detection** (0-2 pts) - Identifies frame where dispersal begins
2. **Biofilm Growth Curve** (0-2 pts) - Quantifies biofilm area over time
3. **Dispersed Cell Quantification** (0-2 pts) - Counts dispersed cells per frame

### Advanced Features (3 points):
4. **Dispersed Cell Characterization** (0-1 pt) - Individual cell detection + morphology
5. **Biofilm Characterization** (0-1 pt) - Yeast/hyphal distribution, length, branching
6. **Cell Tracking** (0-1 pt) - Track cells across 20-hour time course

---

## 🏗️ Architecture

### Stage 1: Instance Segmentation (YOLOv8)
- **Model**: YOLOv8-seg (instance segmentation)
- **Input**: 1392×1040 or 244×242 microscopy images
- **Output**: Cell masks + 7-class classification
- **Classes**:
  - Single dispersed cell
  - Clump dispersed cell
  - Planktonic
  - Yeast form
  - Pseudohyphae
  - Hyphae
  - Biofilm

### Stage 2: Temporal Tracking (ByteTrack)
- **Algorithm**: ByteTrack (state-of-the-art MOT)
- **Input**: Detections across 41 frames
- **Output**: Cell trajectories, growth curves, dispersal events

### Stage 3: Morphological Analysis
- **Methods**: Skeleton extraction, geometric analysis
- **Output**: Cell dimensions, hyphal length, branching patterns

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup Environment

```bash
# Navigate to ml_pipeline directory
cd ml_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check Ultralytics
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"
```

---

## 🚀 Quick Start

### 1. Prepare Data

Convert CVAT XML annotations to YOLO format:

```bash
cd scripts/data_prep

# Convert all datasets
python convert_all_annotations.py \
    --data-dir /Users/aineshmohan/Documents/mlm \
    --output-dir ../../data/processed/yolo_dataset
```

**Output Structure:**
```
data/processed/yolo_dataset/
├── dataset.yaml          # YOLO configuration
├── images/
│   ├── train/           # Training images (164 frames)
│   └── val/             # Validation images (41 frames)
└── labels/
    ├── train/           # Training annotations
    └── val/             # Validation annotations
```

### 2. Train Model

```bash
cd scripts/training

# Train YOLOv8n (nano - fastest)
python train_yolo_segmentation.py \
    --data ../../data/processed/yolo_dataset/dataset.yaml \
    --model n \
    --img-size 640 \
    --batch-size 16 \
    --epochs 100 \
    --device 0

# For better accuracy, use YOLOv8s or YOLOv8m
python train_yolo_segmentation.py \
    --data ../../data/processed/yolo_dataset/dataset.yaml \
    --model s \
    --img-size 640 \
    --batch-size 8 \
    --epochs 150 \
    --device 0
```

**Training on Cloud:**

```bash
# AWS SageMaker / GCP Compute Engine
# Request GPU instance (V100 or A100)
# Estimated training time: 10-20 hours for 100 epochs
```

### 3. Run Inference

Analyze a time-series TIFF file:

```bash
cd scripts/inference

python inference_pipeline.py \
    --model ../../results/candida_segmentation/exp1/weights/best.pt \
    --input /Users/aineshmohan/Documents/mlm/Annotated\ Data/training.tif \
    --output ../../results/analysis_training \
    --conf 0.25 \
    --device 0
```

**Outputs:**
- `results.json` - Complete analysis results
- `cell_counts.csv` - Cell counts per frame
- `biofilm_growth.csv` - Biofilm growth curve
- `dispersed_cells.csv` - Dispersed cell counts
- `analysis_plots.png` - Visualization of key metrics
- `visualizations/` - Annotated frames with tracks

---

## 📊 Expected Performance

### Segmentation Metrics (Target)
- **mAP50**: >0.70 (vs. baseline 0.115)
- **mAP50-95**: >0.50
- **Precision**: >0.60
- **Recall**: >0.65

### Tracking Metrics
- **MOTA**: >0.60
- **IDF1**: >0.70
- **Track completeness**: >80%

---

## 🔬 Dataset Details

### Training Data
- **Total sequences**: 5 annotated
- **Total frames**: 205 (5 × 41 frames)
- **Total annotations**: 7,462 tracked objects
- **Resolution**: 1392×1040 (high-res), 244×242 (low-res)
- **Temporal resolution**: 30 minutes per frame, 20 hours total

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Planktonic | 2,643 | 35.4% |
| Single dispersed | 1,488 | 19.9% |
| Hyphae | 1,306 | 17.5% |
| Clump dispersed | 979 | 13.1% |
| Yeast form | 702 | 9.4% |
| Biofilm | 189 | 2.5% |
| Pseudohyphae | 155 | 2.1% |

**Challenge**: Severe class imbalance (35% planktonic vs. 2% pseudohyphae)

**Solution**: Weighted loss, focal loss, heavy augmentation

---

## 🛠️ Workflow Details

### Data Preparation Pipeline

1. **Unzip annotations** from CVAT XML archives
2. **Parse XML** to extract:
   - Ellipses (individual cells)
   - Polygons (biofilm regions)
   - Polylines (hyphae)
3. **Convert to YOLO format**:
   - Ellipses → polygon approximations
   - Polylines → width-augmented polygons
   - Normalize coordinates (0-1)
4. **Extract TIFF frames** to individual images
5. **Organize dataset** into train/val splits

### Training Pipeline

1. **Data Augmentation** (critical for small dataset):
   - Geometric: rotation (360°), flip, shift, scale
   - Intensity: brightness, contrast, gamma
   - Blur: Gaussian, median, motion
   - Noise: Gaussian, ISO noise
   - CLAHE: contrast enhancement
   - Mosaic & MixUp: multi-image composition

2. **Model Configuration**:
   - Pretrained COCO weights (transfer learning)
   - AdamW optimizer
   - Cosine learning rate schedule
   - Early stopping (patience=50)

3. **Class Balancing**:
   - Weighted loss by inverse frequency
   - Oversampling rare classes

4. **Experiment Tracking**:
   - MLflow for metrics logging
   - Automatic checkpointing
   - Visualization generation

### Inference Pipeline

1. **Load trained model**
2. **For each frame**:
   - Run YOLOv8 detection
   - Extract masks + class predictions
3. **Temporal tracking**:
   - ByteTrack association
   - Handle occlusions & re-identifications
4. **Analysis**:
   - Count cells by type
   - Measure biofilm area
   - Detect dispersal events
5. **Generate outputs**:
   - CSV time-series data
   - JSON structured results
   - Visualization plots

---

## 📈 Competition Deliverables

### How to Extract Each Deliverable

#### 1. Dispersal Initiation (Frame Number)

```python
# From results.json
with open('results/analysis_training/results.json', 'r') as f:
    results = json.load(f)
    dispersal_frame = results['dispersal_initiation_frame']
    print(f"Dispersal begins at frame: {dispersal_frame}")
```

#### 2. Biofilm Growth Curve

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load growth data
df = pd.read_csv('results/analysis_training/biofilm_growth.csv')

# Plot
plt.plot(df['frame'], df['biofilm_area'])
plt.xlabel('Frame')
plt.ylabel('Biofilm Area (pixels)')
plt.title('Biofilm Growth Over Time')
plt.savefig('biofilm_growth.png')
```

#### 3. Dispersed Cell Count

```python
# Load dispersed cell data
df = pd.read_csv('results/analysis_training/dispersed_cells.csv')

# Get count at specific frame
frame_20_count = df[df['frame'] == 20]['dispersed_cells'].values[0]
print(f"Dispersed cells at frame 20: {frame_20_count}")
```

#### 4-6. Advanced Features

All available in `results.json`:
- Cell trajectories: Track individual cells
- Morphological features: From masks
- Classification: 7-class predictions

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python train_yolo_segmentation.py --batch-size 4

# Reduce image size
python train_yolo_segmentation.py --img-size 512
```

### Low Validation Performance

- **Increase epochs**: `--epochs 200`
- **Use larger model**: `--model m` or `--model l`
- **Adjust augmentation**: Modify `utils/augmentation/augmentation_pipeline.py`
- **Check data**: Verify annotations are correct

### Tracking Issues

- **Adjust thresholds** in `cell_tracker.py`:
  - `track_thresh`: Lower for more detections (0.3-0.5)
  - `match_thresh`: IoU threshold (0.7-0.9)
  - `track_buffer`: Frames to keep lost tracks (20-50)

---

## 📝 Team Workflow

### Division of Labor (5 Members)

**Phase 1: Data Preparation (Week 1)**
- Member 1-2: Run annotation conversion, verify output
- Member 3: Set up cloud compute (AWS/GCP)
- Member 4-5: Data exploration, augmentation testing

**Phase 2: Model Training (Week 2-3)**
- Member 1-2: Train models, tune hyperparameters
- Member 3: Monitor training, experiment tracking
- Member 4-5: Validation, error analysis

**Phase 3: Analysis (Week 4)**
- Member 1: Tracking pipeline
- Member 2: Morphological analysis
- Member 3-4: Deliverables extraction
- Member 5: Visualization, presentation

**Phase 4: Final Submission (Week 5)**
- All: Results compilation, documentation, presentation prep

---

## 📚 References

### Papers
- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **ByteTrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2021)
- **Instance Segmentation**: He et al., "Mask R-CNN" (2017)

### Resources
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Albumentations](https://albumentations.ai/)

---

## 🤝 Contributing

Team members should:
1. Create feature branches: `git checkout -b feature/your-feature`
2. Commit regularly with clear messages
3. Push to GitHub: `git push origin feature/your-feature`
4. Update documentation as needed

---

## 📄 License

Internal team use - UW Madison MLM 2025

---

## ✅ Checklist

### Before Training
- [ ] Data converted to YOLO format
- [ ] Dataset YAML verified
- [ ] GPU access confirmed
- [ ] MLflow setup (optional)

### During Training
- [ ] Monitor loss curves
- [ ] Check validation metrics
- [ ] Save best checkpoint
- [ ] Log experiments

### After Training
- [ ] Validate on test set (MattLines9)
- [ ] Run inference pipeline
- [ ] Extract all deliverables
- [ ] Generate visualizations
- [ ] Prepare presentation

### Final Submission
- [ ] Dispersal initiation frame
- [ ] Biofilm growth curve CSV/plot
- [ ] Dispersed cell counts CSV/plot
- [ ] Model weights archived
- [ ] Code documentation complete
- [ ] Presentation slides ready

---

## 📧 Contact

Brain Buddies Team - UW Madison MLM 2025

For questions or issues, refer to project TODO.md or team discussions.

---

**Good luck! 🚀**
