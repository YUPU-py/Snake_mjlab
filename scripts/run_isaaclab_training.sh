#!/bin/bash
# IsaacLab Snake Training Launch Script (with WandB)

# === WandB Configuration ===
export WANDB_API_KEY="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"
export WANDB_ENTITY="ypu900054-university-of-science-and-technology-beijing"
export WANDB_PROJECT="isaaclab-snake-velocity"

# === CUDA Configuration ===
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib/nvidia:$LD_LIBRARY_PATH"

# === Environment ===
CONDA_ENV="snake_train"
CONDA_BIN="conda"
PROJECT_DIR="/home/admin01/workspace_pyx/Snake_Project-main"

# === Training Parameters ===
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-5000}
SEED=${SEED:-42}
GPU_ID=${GPU_ID:-0}

# === Run Name (timestamp by default, override with --run_name) ===
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}

# === Activate Conda ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# === Run Training ===
cd "$PROJECT_DIR"

python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --seed "$SEED" \
    --experiment_name snake_velocity_flat_tracking \
    --run_name "$RUN_NAME" \
    --logger wandb \
    --log_project_name "$WANDB_PROJECT" \
    --headless \
    --device cuda:${GPU_ID} \
    "$@"

echo "Training completed!"
