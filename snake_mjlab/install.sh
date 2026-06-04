#!/bin/bash
# ==============================================================================
# Snake_mjlab 一键安装脚本
# ==============================================================================
# 依赖: mjlab (包含 mujoco, torch, warp-lang 等), Python >= 3.10
# 用法: bash install.sh [--env-name ENV_NAME]
#
set -euo pipefail

# --- 参数解析 ---
ENV_NAME="snake_mjlab"
PYTHON_VERSION="3.11"
CUDA_VERSION="cu126"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)  ENV_NAME="$2"; shift 2 ;;
        --python)    PYTHON_VERSION="$2"; shift 2 ;;
        *)           echo "未知参数: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Snake_mjlab 安装脚本"
echo "=========================================="
echo "  环境名:      $ENV_NAME"
echo "  Python:      $PYTHON_VERSION"
echo "  项目目录:     $PROJECT_DIR"
echo "=========================================="

# --- 0. 检查系统依赖 ---
echo "[0/5] 检查系统依赖..."

command -v conda &>/dev/null || { echo "错误: 请先安装 Miniconda (https://docs.conda.io/en/latest/miniconda.html)"; exit 1; }

# --- 1. 创建 conda 环境 ---
echo "[1/5] 创建 conda 环境 ($ENV_NAME)..."
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  环境 $ENV_NAME 已存在，跳过创建。"
else
    conda create -y -n "$ENV_NAME" python="${PYTHON_VERSION}" -c conda-forge
fi

conda activate "$ENV_NAME"

# --- 2. 安装 PyTorch (CUDA 版本) ---
echo "[2/5] 安装 PyTorch..."

if [[ "$CUDA_VERSION" == "cu126" ]]; then
    # CUDA 12.6
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
elif [[ "$CUDA_VERSION" == "cu128" ]]; then
    # CUDA 12.8
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    # CPU 或其他
    pip install torch torchvision
fi

# --- 3. 安装 mjlab ---
echo "[3/5] 安装 mjlab..."
pip install mjlab

# --- 4. 安装 snake_mjlab 包 ---
echo "[4/5] 安装 snake_mjlab 包..."
cd "$PROJECT_DIR"
pip install -e ./snake_mjlab

# --- 5. 验证安装 ---
echo "[5/5] 验证安装..."
python -c "
import mjlab
import mujoco
import torch
print(f'  mjlab:      {mjlab.__version__}')
print(f'  mujoco:     {mujoco.__version__}')
print(f'  torch:      {torch.__version__}')
print(f'  CUDA:       {\"可用\" if torch.cuda.is_available() else \"不可用\"}')

# 验证 snake_mjlab 导入
from snake_mjlab import SnakeVelocityFlatEnvCfg
from snake_mjlab.rl_cfg import SnakeVelocityFlatPPORunnerCfg
print('  snake_mjlab: OK')
print()
print('安装完成!')
print()
echo '运行训练:'
echo '  conda activate $ENV_NAME'
echo '  python -m mjlab.scripts.train Mjlab-Velocity-Flat-Snake-14DOF'
"
