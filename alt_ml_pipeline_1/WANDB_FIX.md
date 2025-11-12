# WandB Conflict Resolution

## The Problem

You encountered this error when trying to train:

```
wandb: ERROR Invalid project name '/Users/aineshmohan/mlm_outputs/alt_pipeline_1/models/individual':
cannot contain characters '/,\\,#,?,%,:', found '/'
```

## Why This Happens

### Understanding the Architecture

**MLflow and WandB are SEPARATE systems** - they don't interact:

```
┌─────────────────────────────────────────┐
│  Ultralytics YOLOv8                     │
│                                         │
│  Built-in integrations:                 │
│  ├─ TensorBoard (always enabled)       │
│  ├─ WandB (enabled by default) ❌      │
│  ├─ MLflow (enabled by default) ✅     │
│  ├─ ClearML (enabled by default)       │
│  └─ ... other trackers                 │
└─────────────────────────────────────────┘

Our choice: Use ONLY MLflow (disable others)
```

### The Root Cause

1. **Ultralytics has WandB enabled by default** in its settings
2. When you call `model.train(project='/path/to/save')`, YOLO:
   - Uses `project` as a **file path** to save model checkpoints ✅
   - Also passes `project` to **WandB as a project name** ❌
3. WandB rejects the path because project names can't contain `/`

### Why Not Use WandB?

- ❌ Requires cloud account and login
- ❌ Uploads data to cloud (privacy concerns)
- ❌ Needs internet connection
- ❌ Adds complexity

**MLflow is better for this project:**
- ✅ Local storage (no cloud)
- ✅ No account needed
- ✅ Works offline
- ✅ Simple setup

## The Solution

We disable WandB in **three ways** (belt and suspenders approach):

### 1. Environment Variables (lines 16-17)

```python
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
```

This tells WandB library to not initialize at all.

### 2. Ultralytics Settings (line 31)

```python
from ultralytics import settings
settings['wandb'] = False
```

This tells Ultralytics to skip WandB integration callbacks.

### 3. Timing is Critical

These must be set **BEFORE** importing YOLO:

```python
# ✅ CORRECT ORDER:
os.environ['WANDB_DISABLED'] = 'true'  # Set env vars first
from ultralytics import YOLO, settings  # Then import
settings['wandb'] = False               # Then disable in settings

# ❌ WRONG ORDER:
from ultralytics import YOLO            # Too late!
os.environ['WANDB_DISABLED'] = 'true'  # WandB already initialized
```

## Verification

Test that WandB is properly disabled:

```bash
cd alt_ml_pipeline_1
source ~/venvs/alt_pipeline_1/bin/activate

python -c "
import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
from ultralytics import settings
settings['wandb'] = False
print(f'WandB: {settings[\"wandb\"]}')
"
```

Expected output:
```
WandB: False
✓ Success
```

## What You'll See During Training

With WandB disabled, you'll see:

```
TensorBoard: Start with 'tensorboard --logdir /Users/aineshmohan/mlm_outputs/...'
✓ No WandB initialization messages
✓ No WandB errors
✓ Training proceeds normally
```

## Using MLflow Instead

After training completes:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow

# Open browser to http://localhost:5000
```

You'll see:
- All training runs
- Metrics (F1, mAP50, precision, recall)
- Model artifacts
- Comparison between runs

## File Locations

**Model checkpoints** (saved locally):
```
~/mlm_outputs/alt_pipeline_1/models/individual/baseline_fold_0/
├── weights/
│   ├── best.pt        # Best model
│   └── last.pt        # Last epoch
└── results.csv         # Training metrics
```

**MLflow tracking** (local database):
```
~/mlm_outputs/alt_pipeline_1/experiments/mlflow/
└── mlruns/
    └── [experiment_id]/
        └── [run_id]/
            ├── metrics/
            ├── params/
            └── artifacts/
```

**TensorBoard logs** (also created by YOLO):
```
~/mlm_outputs/alt_pipeline_1/models/individual/baseline_fold_0/
└── events.out.tfevents.*
```

## If You Still See WandB Errors

### Option 1: Uninstall WandB (Nuclear Option)

```bash
pip uninstall wandb -y
```

This completely removes WandB, making it impossible to initialize.

### Option 2: Create wandb config file

Create `~/.config/wandb/settings`:
```
[default]
mode = disabled
```

### Option 3: Login to WandB (Not Recommended)

If you want to use WandB instead of MLflow:

```bash
wandb login
# Remove environment variables from train_baseline.py
# Remove settings['wandb'] = False line
```

But we strongly recommend using MLflow for this project.

## Summary

**Problem**: YOLO tries to use file path as WandB project name
**Root Cause**: WandB enabled by default in Ultralytics
**Solution**: Disable WandB in environment and settings
**Alternative**: Use MLflow (what we're doing)

**Key Takeaway**: MLflow and WandB are independent - we chose MLflow and disabled WandB to avoid conflicts.

---

**Status**: ✅ Fixed in commit (disabled WandB, using MLflow exclusively)
