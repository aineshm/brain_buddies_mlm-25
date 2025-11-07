# Quick Start Guide

**Get running in 5 minutes!**

---

## 🚀 Express Setup

### 1. Install Dependencies (5 min)

```bash
cd ml_pipeline
pip install -r requirements.txt
```

### 2. Prepare Data (10 min)

```bash
cd scripts/data_prep

python3 convert_all_annotations.py \
    --data-dir /Users/aineshmohan/Documents/mlm \
    --output-dir ../../data/processed/yolo_dataset

# Expected output:
# Training images: 164
# Validation images: 41
# Total images: 205
```

### 3. Train Model (10-20 hours on GPU)

```bash
cd ../training

# Quick training (for testing)
python3 train_yolo_segmentation.py \
    --data ../../data/processed/yolo_dataset/dataset.yaml \
    --model n \
    --epochs 50 \
    --batch-size 16 \
    --device 0

# Production training (better results)
python3 train_yolo_segmentation.py \
    --data ../../data/processed/yolo_dataset/dataset.yaml \
    --model s \
    --epochs 150 \
    --batch-size 8 \
    --device 0
```

**⚠️ No GPU?** Use Google Colab or request AWS credits.

### 4. Run Analysis (5 min)

```bash
cd ../inference

python3 inference_pipeline.py \
    --model ../../results/candida_segmentation/*/weights/best.pt \
    --input /Users/aineshmohan/Documents/mlm/Annotated\ Data/training.tif \
    --output ../../results/analysis_training \
    --device 0
```

### 5. View Results

```bash
# Check key deliverables
cat ../../results/analysis_training/results.json | grep "dispersal_initiation_frame"

# Open plots
open ../../results/analysis_training/analysis_plots.png

# View video
# (Optional: create video from visualizations/)
```

---

## 📊 Extract Competition Deliverables

### Python Script

```python
import json
import pandas as pd

# Load results
with open('results/analysis_training/results.json', 'r') as f:
    results = json.load(f)

# 1. Dispersal Initiation (0-2 points)
dispersal_frame = results['dispersal_initiation_frame']
print(f"✓ Dispersal begins at frame: {dispersal_frame}")

# 2. Biofilm Growth Curve (0-2 points)
df_biofilm = pd.read_csv('results/analysis_training/biofilm_growth.csv')
print(f"✓ Biofilm growth data: {len(df_biofilm)} time points")
print(f"  Initial area: {df_biofilm['biofilm_area'].iloc[0]:.0f} pixels")
print(f"  Final area: {df_biofilm['biofilm_area'].iloc[-1]:.0f} pixels")

# 3. Dispersed Cell Count (0-2 points)
df_dispersed = pd.read_csv('results/analysis_training/dispersed_cells.csv')
print(f"✓ Dispersed cell counts: {len(df_dispersed)} time points")
print(f"  Peak count: {df_dispersed['dispersed_cells'].max()} cells")

# 4-6. Advanced features (0-3 points)
print(f"✓ Total cell tracks: {results['total_tracks']}")
print(f"✓ Average track length: {results['track_statistics']['avg_track_length']:.1f} frames")
print("✓ Cell characterization: Available from masks")
```

---

## 🎯 Common Tasks

### Validate Model

```bash
cd scripts/training

python3 train_yolo_segmentation.py \
    --data ../../data/processed/yolo_dataset/dataset.yaml \
    --validate-only \
    --weights ../../results/candida_segmentation/*/weights/best.pt
```

### Analyze Different Sample

```bash
cd scripts/inference

# MattLines27 (validation set)
python3 inference_pipeline.py \
    --model ../../results/candida_segmentation/*/weights/best.pt \
    --input /Users/aineshmohan/Documents/mlm/Annotated\ Data/MattLines27.tif \
    --output ../../results/analysis_MattLines27

# MattLines9 (test set - no annotations)
python3 inference_pipeline.py \
    --model ../../results/candida_segmentation/*/weights/best.pt \
    --input /Users/aineshmohan/Documents/mlm/Annotated\ Data/MattLines9.tif \
    --output ../../results/analysis_MattLines9
```

### Adjust Detection Thresholds

```bash
# Higher confidence = fewer detections (more precise)
python3 inference_pipeline.py \
    --model ... \
    --input ... \
    --output ... \
    --conf 0.5

# Lower confidence = more detections (higher recall)
python3 inference_pipeline.py \
    --model ... \
    --input ... \
    --output ... \
    --conf 0.15
```

---

## 🐍 Python API Usage

```python
from scripts.inference.inference_pipeline import CandidaAnalysisPipeline

# Create pipeline
pipeline = CandidaAnalysisPipeline(
    model_path='results/candida_segmentation/exp1/weights/best.pt',
    confidence_threshold=0.25,
    device='0'
)

# Run analysis
results = pipeline.process_sequence(
    tiff_path='/path/to/sequence.tif',
    output_dir='results/my_analysis',
    save_visualizations=True
)

# Access results
print(f"Dispersal at frame: {results['dispersal_initiation_frame']}")
print(f"Total tracks: {results['total_tracks']}")

# Per-frame data
for frame, count in results['cell_counts_per_frame'].items():
    print(f"Frame {frame}: {count} cells")
```

---

## 🔧 Troubleshooting Quick Fixes

### Issue: Import Errors

```bash
# Make sure you're in the right directory
cd ml_pipeline

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python3 train_yolo_segmentation.py --batch-size 4

# Or use CPU (slow!)
python3 train_yolo_segmentation.py --device cpu
```

### Issue: Model Not Found

```bash
# Find your model
find results/ -name "best.pt"

# Use full path
python3 inference_pipeline.py \
    --model /full/path/to/best.pt \
    ...
```

### Issue: No Detections

```bash
# Lower confidence threshold
python3 inference_pipeline.py --conf 0.1

# Check if model trained properly
python3 train_yolo_segmentation.py --validate-only --weights ...
```

---

## 📦 File Organization

After running pipeline:

```
ml_pipeline/
├── data/
│   └── processed/
│       └── yolo_dataset/
│           ├── dataset.yaml         ✓ Dataset config
│           ├── images/train/        ✓ 164 training images
│           ├── images/val/          ✓ 41 validation images
│           └── labels/              ✓ Annotations
├── results/
│   ├── candida_segmentation/
│   │   └── exp1/
│   │       ├── weights/best.pt     ✓ Trained model
│   │       └── results.csv         ✓ Training metrics
│   └── analysis_training/
│       ├── results.json            ✓ All results
│       ├── cell_counts.csv         ✓ Deliverable #3
│       ├── biofilm_growth.csv      ✓ Deliverable #2
│       ├── dispersed_cells.csv     ✓ Deliverable #3
│       ├── analysis_plots.png      ✓ Visualization
│       └── visualizations/         ✓ Frame-by-frame
└── scripts/                        ✓ All code
```

---

## ⏱️ Timeline Estimate

| Task | Time | Depends On |
|------|------|------------|
| Install dependencies | 5 min | - |
| Convert annotations | 10 min | Dependencies |
| Train YOLOv8n (quick) | 4 hours | GPU access |
| Train YOLOv8s (production) | 15 hours | GPU access |
| Run inference | 5 min | Trained model |
| Extract deliverables | 2 min | Inference results |

**Total**: ~20 hours (mostly GPU training)

---

## 🎓 Learning Resources

### For Team Members New to ML

1. **YOLOv8 Basics**: https://docs.ultralytics.com/
2. **Object Tracking**: https://github.com/ifzhang/ByteTrack
3. **PyTorch Tutorial**: https://pytorch.org/tutorials/

### Recommended Reading Order

1. Read [README.md](README.md) for full context
2. Follow this Quick Start
3. Explore notebooks/ for examples (if created)
4. Dive into code in scripts/

---

## 💡 Pro Tips

1. **Start small**: Train on subset first (10 epochs) to verify pipeline
2. **Monitor training**: Watch MLflow or TensorBoard
3. **Save checkpoints**: Don't lose progress to crashes
4. **Test early**: Run inference on 1 frame before full sequence
5. **Document**: Keep notes on what hyperparameters work

---

## 🆘 Getting Help

1. Check [README.md](README.md) Troubleshooting section
2. Review error messages carefully
3. Google the error + "ultralytics" or "pytorch"
4. Ask team members
5. Check Ultralytics docs

---

## ✅ Success Criteria

You're ready for final submission when:

- [ ] Model trained to >0.7 mAP50
- [ ] Inference runs on all test samples
- [ ] All 3 minimal deliverables extracted
- [ ] Visualizations look reasonable
- [ ] Code is documented
- [ ] Results are reproducible

---

**You've got this! 🚀**

Now go train that model and win the competition!
