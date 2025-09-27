# Backdoor-attack-and-defense-methods-for-GNN
A list of papers, codes, and resources on Graph Neural Network (GNN) backdoor attacks and defenses.

---

## Contents
- [Attack Methods](#attack-methods)
- [Defense Methods](#defense-methods)
- [Structure](#structure)
- [Contributing](#contributing)
- [datasets](#静态图)
- [datasets](#动态图)
- [Experimental results](#整理结果)
- [Experimental results](#复现结果)
- [Citation](#citation)
---

# Attack Methods

> 表格按：年份 | 任务层级 | 方法名 | 论文链接 | 代码链接

| Year | Level | Method | Paper | Code |
|------|-------|--------|-------|------|
| 2021 | Graph-level | Backdoor Attacks to Graph Neural Networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3450569.3463560) | [Code](https://github.com/zaixizhang/graphbackdoor) |
| 2021 | Graph-level | Backdoor attack of graph neural networks based on subgraph trigger | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-92638-0_17) | No code|
| 2021 | Hybrid-level | Graph backdoor | [Paper](https://www.usenix.org/system/files/sec21-xi.pdf) | [Code](https://github.com/zhaohan-xi/GraphBackdoor) |
| 2021 | Hybrid-level  | Explainability-based backdoor attacks against graph neural networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3468218.3469046) |No code|
| 2022 | Node-level  | A general backdoor attack to graph neural networks based on explanation method | [Paper](https://ieeexplore.ieee.org/abstract/document/10063438) | No code |
| 2022 | Graph-level | Poster: Clean-label Backdoor Attack on Graph Neural Networks| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002000) | No code |
| 2022 | Graph-level | Transferable Graph Backdoor Attack| [Paper](https://arxiv.org/pdf/2207.00425) |No code |
| 2022 | Graph-level | More is better (mostly): On the backdoor attacks in federated graph neural networks| [Paper](https://dl.acm.org/doi/abs/10.1145/3564625.3567999) |  [Code](https://github.com/xujing1994/bkd_fedgnn)  |
| 2023 | Node-level  | Feature-Based Graph Backdoor Attack in the Node Classification Task | [Paper](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2023/5418398) | No code |
| 2023 | Node-level  | Graph contrastive backdoor attacks | [Paper](https://proceedings.mlr.press/v202/zhang23e/zhang23e.pdf) | No code |
| 2023 | Node-level  | Unnoticeable Backdoor Atacks on Graph Neural Networks | [Paper](https://dl.acm.org/doi/10.1145/3543507.3583392) | [Code](https://github.com/ventr1c/UGBA) |
| 2023 | Node-level  | A Clean-graph Backdoor Attack against Graph Convolutional Networks with Poisoned Label Only | [Paper](https://arxiv.org/pdf/2404.12704) | No code |
| 2023 | Node-level  | PerCBA: Persistent Clean-label Backdoor Attacks on Semi-Supervised Graph Node Classification | [Paper](https://ceur-ws.org/Vol-3505/paper_4.pdf) |No code |
| 2023 | Node-level  | Effective Backdoor Attack on Graph Neural Networks in Spectral Domain | [Paper](https://ieeexplore.ieee.org/abstract/document/10318195) | No code |
| 2023 | Node-level  | Poster: Multi-target & Multi-trigger Backdoor Attacks on Graph Neural Networks | [Paper](https://repository.ubn.ru.nl/handle/2066/299150) | No code  |
| 2023 | Graph-level | Motif-backdoor: Rethinking the backdoor attack on graph neural networks via motif| [Paper](https://arxiv.org/pdf/2210.13710) |  [Code](https://github.com/Seaocn/Motif-Backdoor)  |
| 2023 | Link-prediction | Link-backdoor: Backdoor attack on link prediction via node injection| [Paper](https://ieeexplore.ieee.org/abstract/document/10087329) |  [Code](https://github.com/Seaocn/Link-Backdoor)  |
| 2023 | Link-prediction | Dyn-Backdoor: Backdoor Attack on Dynamic Link Prediction| [Paper](https://arxiv.org/pdf/2110.03875) |  No code |
| 2023 | Hybrid-level  | Bkd-fedgnn: A benchmark for classification backdoor attacks on federated graph neural network | [Paper](https://arxiv.org/pdf/2306.10351) | [Code](https://github.com/usail-hkust/BkdFedGCN) |
| 2024 | Node-level  | Multi-target label backdoor attacks on graph neural networks | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002000) |No code |
| 2024 | Node-level  | A clean-label graph backdoor attack method in node classification task | [Paper](https://arxiv.org/pdf/2401.00163) |No code |
| 2024 | Graph-level | Explanatory subgraph attacks against Graph Neural Networks| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608024000030) |No code |
| 2024 | Graph-level | Unveiling the threat: Investigating distributed and centralized backdoor attacks in federated graph neural network| [Paper](https://dl.acm.org/doi/full/10.1145/3633206) | No code |
| 2024 | Graph-level | Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses| [Paper](https://dl.acm.org/doi/abs/10.1145/3658644.3690187) |  [Code](https://github.com/Yuxin104/Opt-GDBA)  |
| 2024 | Link-prediction | A backdoor attack against link prediction tasks with graph neural networks| [Paper](https://arxiv.org/pdf/2401.02663) |  No code |
| 2024 | Hybrid-level | 自适应图神经网络后门攻击方法研究 | 西电硕士论文 | No code |
| 2025 | Node-level | Backdoor Graph Condensation  | [Paper](https://arxiv.org/pdf/2407.11025) | [Code](https://github.com/JiahaoWuGit/BGC) |
| 2025 | Node-level | Backdoor Attack on Vertical Federated Graph Neural Network Learning  | [Paper](https://arxiv.org/pdf/2410.11290?)| No code |
| 2025 | Hybrid-level | Adaptive Backdoor Attacks with Reasonable Constraints on Graph Neural Networks  | [Paper](https://arxiv.org/pdf/2503.09049) | No code |
| 2025 | Link-prediction | Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems  | [Paper](https://arxiv.org/pdf/2506.08401?) | No code|


---

# Defense Methods

> 表格按：年份 | 防御阶段 | 方法名 | 论文链接 | 代码链接

| Year | Stage | Method | Paper | Code |
|------|-------|--------|-------|------|
| 2021 | Testing-time | Backdoor Attacks to Graph Neural Networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3450569.3463560) | [Code](https://github.com/zaixizhang/graphbackdoor) |
| 2022 | Testing-time | 图神经网络后门攻击及防御机制研究 | 中南大学2022陈榕 | No code |
| 2022 | Testing-time | Defending against backdoor attack on graph neural network by explainability | [Paper](https://arxiv.org/pdf/2209.02902) |No code |
| 2023 | Training-time | XGBD: Explanation-guided Graph Backdoor Detection | [Paper](https://arxiv.org/pdf/2308.04406) |  [Code](https://github.com/GuanZihan/GNN_backdoor_detection) |
| 2023 | Training-time | 基于对比学习的图神经网络后门攻击防御方法 | 通信学报 | No code |
| 2024 | Training-time | Securing GNNs: Explanation-based identification of backdoored training graphs | [Paper](T[ODO_SPECTRAL_PAPER_LINK](https://arxiv.org/pdf/2403.18136v1)) | No code|
| 2024 | Training-time | On the Robustness of Graph Reduction Against GNN Backdoor | [Paper](https://arxiv.org/pdf/2407.02431?) |  [Code](https://github.com/GuanZihan/GNN_backdoor_detection) |
| 2024 | Training-time | DMGNN: Detecting and Mitigating Backdoor Attacks in Graph Neural Networks | [Paper](https://arxiv.org/pdf/2410.14105) | No code |
| 2022 | Testing-time | 面向图神经网络后门攻击的触发器恢复和移除方法 | 西电2024辛柳莹 | No code |
| 2024 | Federated | Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses| [Paper](https://dl.acm.org/doi/abs/10.1145/3658644.3690187) |  [Code](https://github.com/Yuxin104/Opt-GDBA)  |
| 2024 | Testing-time | Defense-as-a-service: Black-box shielding against backdoored graph models | [Paper](https://arxiv.org/abs/2410.04916) | No code|
| 2025 | Training-time | Robustness Inspired Graph Backdoor Defense | [Paper](https://arxiv.org/pdf/2406.09836) |  [Code](https://github.com/zzwjames/RIGBD) |
| 2025 | Federated | Energy-based backdoor defense against federated graph learnin | [Paper](https://openreview.net/pdf?id=5Jc7r5aqHJ) | [Code](https://github.com/ZitongShi/fedTGE) |
| 2025 | Training-time | DShield: Defending against Backdoor Attacks on Graph Neural Networks via Discrepancy Learning | [Paper](https://www.ndss-symposium.org/wp-content/uploads/2025-798-paper.pdf) |  [Code](https://github.com/csyuhao/DShield) |


---

# Structure

- `README.md` — this file (paper list + links)  
- `papers/` — optional: store PDFs or BibTeX files (upload via "Add file → Upload files")  
- `code/` — optional: if you include any local example code (or link out to external repos)  
- `scripts/` — optional: download / reproduce scripts

---

#静态图
| name | Level |
|------|-------|
| [Bitcoin](https://github.com/zaixizhang/graphbackdoor/blob/main/dataset.zip) | Graph-level |
| [COLLAB](https://github.com/zaixizhang/graphbackdoor/blob/main/dataset.zip) | Graph-level |
| [MUTAG](https://github.com/zaixizhang/graphbackdoor/blob/main/dataset.zip) | Graph-level |
| [twitter](https://github.com/zaixizhang/graphbackdoor/blob/main/dataset.zip) | Graph-level |
| [DD](https://networkrepository.com/DD.php) | Graph-level |
| [Mutagenicity](https://networkrepository.com/Mutagenicity.php) | Graph-level |
| [Proteins-full](https://networkrepository.com/PROTEINS-full.php) | Graph-level |
| [NCI1](https://chrsmrrs.github.io/datasets/docs/datasets) | Graph-level |
---
#动态图
| name | Level |
|------|-------|
| [Radoslaw](http://konect.uni-koblenz.de/networks/radoslaw%20email) | Link-Prediction |
| [Contact](http://konect.uni-koblenz.de/networks/contact) | Link-Prediction |
| [Fb-forum](http://konect.uni-koblenz.de/networks/facebook-wosn-wall) | Link-Prediction |
| [DNC](http://konect.uni-koblenz.de/networks/dnc-temporalGraph) | Link-Prediction |
---
#Experimental results
## GNN Backdoor Results (ASR/CAD, %)
<div style="overflow-x:auto;">

| 任务类型 | 后门攻击方法 | Cora | CiteSeer| PubMed | MUTAG | Mutagenicity | NCI1 | PROTEINS |
|---------|-------------|----------|-------------|------------|-----------|-----------------|----------|-------------|
| 节点分类 | General-Backdoor(34) | - | - | - | - | - | - | - |
| 节点分类 | NFTA(35) | - | 83.48/3.39 | - | - | - | - | - |
| 节点分类 | GCBA(36) | 96.2/0.2 | 94.6/0.2 | - | - | - | - | - |
| 节点分类 | UGBA(37) | 96.95/- | - | 92.27/- | - | - | - | - |
| 节点分类 | CBAG(38) | 99.83/2.41 | 96.8/-1.93 | 98.3/1.38 | - | - | - | - |
| 节点分类 | CGBA(39) | 98/5.6 | 99.2/-2.4 | 89.8/-0.1 | - | - | - | - |
| 节点分类 | PecCBA(40) | 89.6/0.27 | 70.34/2.55 | 91.85/-3.52 | - | - | - | - |
| 节点分类 | Spectrum-Backdoor(41) | 100/1.2 | 98.2/-0.1 | 100/0.1 | - | - | - | - |
| 节点分类 | BGC(33) | 100/- | 100/- | - | - | - | - | - |
| 节点分类 | BVG(42) | 99.86/- | - | 100/- | - | - | - | - |
| 节点分类 | Multi-target & multi-trigger(43) | 98.15/- | 98.51/- | - | - | - | - | - |
| 节点分类 | MLGB(44)| 99.21/-0.55 | - | 99.36/0.33 | - | - | - | - |
| 节点分类 | SBA(45) |- | - | - | - | - | - | - |
| 图分类 | Subgraph-Trigger Backdoor Attack(46) | - | - | - | - | - | 100/- | - |
| 图分类 | CLB(47) | - | - | - |  87.83/0.16 | - | 98.47/0.88 | - |
| 图分类 | ESA(48) | - | - | - | - | -/0.01 | -/0.005/- |-/-0.01 | 
| 图分类 | Motif-backdoor(49) | - | - | - | - | - | 99.72/2.18 | 89.08/-4.45|
| 图分类 | TRAP(50) | - | - | - | - | - | - | 77.78/2.24 |
| 图分类 | CBD、CBDA(51) | - | - | - | - | - | -/2.5 | - |
| 图分类 | CBA、CBDA(52) | - | - | - | - | - | -/2.74 | - |
| 图分类 | Opt-GDBA(53) | - | - | - | 85/- | - | - | 90/- |
| 链接预测 | Link-Backdoor(54) | 92.56/1.08 | 98.78/4.97 | 83.25/2.96 | - | - | - | - |
| 链接预测 | BALP(16) | 99.16/0.07 | 99.94/0.39 | - | - | - | - | - |
| 链接预测 | Dyn-backdoor(55) | - | - | - | - | - | - | - |
| 混合型 | GTA(56) | - | - | - | - | - | - | - |
| 混合型 | EBA(57) | 84.22/1.95 | 96.26/1.72 | - | - | 97.69/2.65 | - | - |
| 混合型 | ABARC(65) | 98.16/1.85 | 99.7/1.81 | - | - | - | - | - |

</div>
## Defense Methods Results (ASR/ACC, %)
<div style="overflow-x:auto;">

| 攻击方法 | 防御方法 | Cora(ASR/ACC) | CiteSeer(ASR/ACC) | PubMed(ASR/ACC) | Physics(ASR/ACC) | Flickr(ASR/ACC) | OGB-arxiv(ASR/ACC) |
|---------|----------|---------------|------------------|-----------------|------------------|-----------------|-------------------|
| GTA | GCN | 98.98/82.58 | 100/73.7 | 93.09/85.13 | 96.42/89.3 | 88.45/41.23 | 75.34/65.76 |
| GTA | GNNGuard[85] | 40.22/78.52 | 55.26/63.55 | 26.93/81.68 | 43.87/78.72 | 0/40.4 | 0.04/62.58 |
| GTA | RobustGCN[86] | 90.46/80.37 | 95.31/95.31 | 93.12/81.68 | 95.47/94.84 | 85.42/42.26 | 70.95/56.08 |
| GTA | Prune[37] | 17.63/83.06 | 12.24/72.46 | 28.1/85.05 | 8.34/88.45 | 12.56/41.27 | 0.01/63.97 |
| GTA | OD[78] | 0.04/83.47 | 0.04/72.84 | 0.03/85.27 | 0.12/90.24 | 2.12/41.42 | 0.01/65.23 |
| GTA | RS[75] | 53.14/73.33 | 52.86/65.66 | 42.28/84.58 | 57.7/92.52 | 52.85/42.31 | 42.72/58.48 |
| GTA | ABL[87] | 19.93/81.85 | 13.09/73.19 | 16.18/83.92 | 16.82/92.06 | 17.54/41.46 | 11.28/63.24 |
| GTA | RIGBD[76] | 0/83.7 | 0.34/74.1 | 0.01/84.32 | 0.32/95.1 | 0/44.21 | 0.01/66.51 |
| GTA | Graph-Reduction[71] | 43.61/65.67 | -/- | 44.43/83.13 | -/- | -/- |14.13/39.72 |
| GTA | DMGNN[73] | 0.5/82.1 | -/- | 0.9/83.6 | -/- | 1.3/46.2 | 0.7/67.1 |
| GTA | DShield[84] | 0.74/81.92 | -/- | 0.73/85.28 | -/- | 0.47/50.26 | 0.92/62.72 |
| UGBA | GCN | 98.76/83.42 | 100/74.7 | 96.42/84.64 | 100/95.94 | 93.14/44.71 | 98.82/63.95 |
| UGBA | GNNGuard | 43.17/78.15 | 94.53/64.76 | 98.97/81.48 | 95.26/87.54 | 96.93/42.15 | 92.51/64.61 |
| UGBA | RobustGCN | 98.67/80 | 99.82/71.69 | 99.9/82.85 | 99.94/94.08 | 32.17/41.82 | 90.35/56.18 |
| UGBA | Prune | 98.89/82.66 | 97.68/74.35 | 92.87/85.09 | 94.67/93.87 | 91.43/43.65 | 93.07/62.58 |
| UGBA | OD | 0.03/83.65 | 0.06/73.8 | 0.01/85.19 | 0.02/95.36 | 1.65/43.57 | 0.01/65.35 |
| UGBA | RS | 54.24/70.37 | 50.34/69.88 | 44.41/84.68 | 47.77/94.29 | 20.69/42.18 | 40.3/58.76 |
| UGBA | ABL | 15.13/81.48 | 12.08/73.19 | 28.6/84.37 | 14.87/94.69 | 15.32/41.66 | 32.26/64.93 |
| UGBA | RIGBD | 0.01/84.81 | 0/73.8 | 0.01/85.13 | 0.12/95.71 | 0/43.66 | 0.01/65.21 |
| UGBA | Graph-Reduction[71] | 35.76/75.29 | -/- | 56.77/84.46 | -/- | -/- | 14.13/39.72 |
| UGBA | DMGNN | 1.1/81.7 | -/- | 1.5/82.4 | -/- | 1.7/44.9 | 0.9/65.8 |
| UGBA | DShield[84] | 1.33/82.15 | -/- | 2.24/84.7 | -/- | 3.49/50.64 | 0/58.32 |
| DPGBA[78] | GCN | 97.72/83.34 | 100/74.09 | 98.63/85.22 | 100/95.59 | 90.79/44.87 | 95.63/65.72 |
| DPGBA[78] | GNNGuard | 85.61/78.52 | 46.98/60.84 | 44.12/80.82 | 88.72/88.76 | 95.85/13.52 | 94.66/62.29 |
| DPGBA[78] | RobustGCN | 96.68/81.11 | 91.64/71.08 | 94.88/82.54 | 91.25/94.87 | 24.6/41.69 | 90.09/60.38 |
| DPGBA[78] | Prune | 91.82/85.28 | 94.8/73.21 | 88.64/85.13 | 94.27/94.73 | 88.96/44.75 | 90.47/65.53 |
| DPGBA[78] | OD | 94.33/83.58 | 98.42/73.66 | 91.32/85.12 | 98.72/95.48 | 90.42/43.63 | 93.3/65.47 |
| DPGBA[78] | RS | 51.29/69.63 | 50.34/71.08 | 48.83/85.44 | 48.19/95.3 | 27.94/42.21 | 41.18/58.44 |
| DPGBA[78] | ABL | 86.72/79.26 | 11.41/73.49 | 45.58/79.45 | 11.74/95.3 | 30.42/41.71 | 52.56/63.88 |
| DPGBA[78] | RIGBD | 0.01/85.19 | 0.33/73.79 | 0.03/84.56 | 0.21/95.79 | 0/43.78 | 0/65.24 |

</div>
## Experimental Results Summary

| 数据集 | 方法 | 评估指标 | 实验结果 | 备注 |
|--------|------|----------|----------|------|
| Cora | UGBA[37] | ASR | 0.97 | |
| Cora | UGBA[37] | BA | 0.8385 | |
| CiteSeer |  |  |  | |
| PubMed | UGBA[37] | ASR | 0.898 | |
| PubMed | UGBA[37] | BA | 0.8509 | |
| OGBN-Arxiv | UGBA[37] | ASR | 0.9837 | |
| OGBN-Arxiv | UGBA[37] | CAD |  | |
| Bitcoin | SBA[45] | ASR | 0.7131 | |
| Bitcoin | SBA[45] | CAD | 0.0197 | |
| Bitcoin | SBA[45] | BA | 0.7077 | |
| Twitter | SBA[45] | ASR | 0.8469 | |
| Twitter | SBA[45] | CAD | 0.0333 | |
| Twitter | SBA[45] | BA | 0.6549 | |
| COLLAB | SBA[45] | ASR | 0.7547 | |
| COLLAB | SBA[45] | CAD | 0.0089 | |
| COLLAB | SBA[45] | BA | 0.7268 | |
| NCI1 | More is better[51] | ASR | 1 | DBA |
| NCI1 | More is better[51] | CAD | 0.01 | DBA |
| NCI1 | More is better[51] | BA | 0.821 | DBA |
| NCI1 | More is better[51] | ASR | 0.813 | CBA |
| NCI1 | More is better[51] | CAD | 0.016 | CBA |
| NCI1 | More is better[51] | BA | 0.815 | CBA |
| MTGA | Opt-GDBA | ASR | 1 | |
| MTGA | Opt-GDBA | BA | 0.7619 | |
# Citation

If you use this list or our survey in your work, please cite:

```bibtex
@article{your_survey_2025,
  title={Backdoor Attacks and Defenses in Graph Neural Networks: A Survey},
  author={Your Name and Coauthors},
  year={2025},
  note={arXiv:xxxx.xxxxx or conference info}
}
