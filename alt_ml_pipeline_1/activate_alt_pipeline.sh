#!/bin/bash
# Activate Alt ML Pipeline 1 environment
source ~/venvs/alt_pipeline_1/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/alt_ml_pipeline_1"
export ALT_PIPELINE_OUTPUT="$HOME/mlm_outputs/alt_pipeline_1"
echo "Alt ML Pipeline 1 environment activated"
echo "Output directory: $ALT_PIPELINE_OUTPUT"
