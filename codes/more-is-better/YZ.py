import torch
import dgl

print("PyTorch CUDA available:", torch.cuda.is_available())

# 创建一个最小图并放到 GPU，如果能成功说明 DGL GPU 正常
g = dgl.graph(([0], [1]))
g = g.to("cuda")
print("DGL graph device:", g.device)
