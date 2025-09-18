#!/usr/bin/env bash

# 跳过Flickr数据集，运行其他数据集的脚本
# 因为Flickr需要从Google Drive下载，网络不可达

echo "Skipping Flickr due to network issues, running other datasets..."

# 创建FX目录
mkdir -p FX/{Cora,Pubmed,Arxiv}

echo "Note: Flickr dataset requires Google Drive access which is currently unavailable."
echo "If you need to run Flickr experiments, please:"
echo "1. Download the files manually from Google Drive:"
echo "   - adj_full.npz: https://drive.google.com/file/d/1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy"
echo "   - feats.npy: https://drive.google.com/file/d/1gJ3F7G_PJwg1WQ0Ff2jdtgSbD1lVGrRr" 
echo "   - class_map.json: https://drive.google.com/file/d/1A6ny2wYqyD7n4SZdMjF0HX0rCx0D4s0e"
echo "2. Place them in data/Flickr/raw/"
echo "3. Then run the original script"

echo "Current available datasets: Cora, Pubmed, OGB-Arxiv"
echo "You can run these individually with run_adaptive.py"