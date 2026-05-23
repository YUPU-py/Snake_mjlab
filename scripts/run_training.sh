#!/bin/bash
# mjlab Snake Training Launch Script

# === WandB Configuration ===
export WANDB_API_KEY="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"
export WANDB_ENTITY="ypu900054-university-of-science-and-technology-beijing"
export WANDB_PROJECT="mjlab"

# === CUDA Configuration ===
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH"

# === Environment ===
CONDA_ENV="snake"
PROJECT_DIR="/home/admin01/workspace_pyx/Snake_Project-main"

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
