# BGC (Backdoor Graph Condensation)
This is the official PyTorch implementation for the [paper](https://arxiv.org/abs/2407.11025):
>Jiahao Wu, Ning Lu, Zeiyu Dai, Kun Wang, Wenqi Fan, Shengcai Liu, Qing Li, Ke Tang. Backdoor Graph Condensation. ICDE 2025.

## Overview
While existing graph condensation studies mainly focus on the best trade-off between graph size and the GNNs' performance (model utility), they overlook the security issues of graph condensation. To bridge this gap, we first explore backdoor attack against the GNNs trained on the condensed graphs. We introduce an effective backdoor attack against graph condensation, termed BGC. This attack aims to (1) preserve the condensed graph quality despite trigger injection, and (2) ensure trigger efficacy through the condensation process, achieving a high attack success rate. Specifically, BGC consistently updates triggers during condensation and targets representative nodes for poisoning.

## Requirements
```
torch==1.7.0
torch_geometric==1.6.3
scipy==1.6.2
numpy==1.19.2
ogb==1.3.0
tqdm==4.59.0
torch_sparse==0.6.9
scikit_learn==1.0.2
```

## Dataset
All the datasets can be downloaded via the package Pytorch Geometric. Therefore, we only provide the Cora in the folders.

## Running
The instructions and hyperparameters are recorded in 0-run-attacks.sh.

## Acknowledgement

Please cite the following paper as the references if you use our codes.

```
@article{BGC2025wu,
  author={Wu, Jiahao and Lu, Ning and Dai, Zeyu and Wang, Kun and Fan, Wenqi and Liu, Shengcai and Li, Qing and Tang, Ke},
  journal={ICDE 2025}, 
  title={Backdoor Graph Condensation}, 
  year={2025}}
```
