#!/bin/bash
# mjlab Snake Training Launch Script

# === WandB Configuration (optional) ===
# export WANDB_API_KEY="your-api-key"
# export WANDB_ENTITY="your-entity"
# export WANDB_PROJECT="mjlab"

# === CUDA Configuration ===
# export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH"

# === Environment ===
CONDA_ENV=${CONDA_ENV:-snake}
PROJECT_DIR=${PROJECT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}

# === Training Parameters (customize as needed) ===
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-20000}
GPU_IDS=${GPU_IDS:-0}

# === Activate Conda ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# === Run Training ===
cd "$PROJECT_DIR"
python -m mjlab.scripts.train \
    Mjlab-Velocity-Flat-Snake-14DOF \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    "$@"

echo "Training completed!"
