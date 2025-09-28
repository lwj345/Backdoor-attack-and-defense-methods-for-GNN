#!/bin/bash
# 文件名：install_all_dependencies.sh

echo "=== 安装所有缺失的依赖 ==="

# 激活环境
conda activate fedgnn

# 清理现有安装
echo "清理现有安装..."
pip uninstall dgl torch torchvision setuptools packaging -y

# 安装兼容的版本组合
echo "安装兼容版本..."
pip install setuptools==65.0.0
pip install packaging
pip install ipdb

# 安装PyTorch
echo "安装PyTorch..."
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

# 安装DGL (CUDA版本)
echo "安装DGL CUDA版本..."
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html

# 安装其他依赖
echo "安装其他依赖..."
pip install networkx PyYAML scikit-learn joblib hdbscan

# 验证安装
echo "验证所有依赖..."
python -c "
import torch
import dgl
import hdbscan
import networkx
import yaml
import sklearn
import joblib
import ipdb
print('所有依赖安装成功！')
"

echo "=== 安装完成 ==="