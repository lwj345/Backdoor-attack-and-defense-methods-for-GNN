# Backdoor-attack-and-defense-methods-for-GNN
A list of papers, codes, and resources on Graph Neural Network (GNN) backdoor attacks and defenses.

---

## Contents
- [Attack Methods](#attack-methods)
- [Defense Methods](#defense-methods)
- [Structure](#structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [datasets](#静态图)
-  [datasets](#动态图)

---

# Attack Methods

> 表格按：年份 | 任务层级 | 方法名 | 论文链接 | 代码链接（有则写，没有写 No official code）

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
| 2024 | Hybrid-level | 自适应图神经网络后门攻击方法研究 | 西电硕士论文 | [Code](https://github.com/zhaohan-xi/GraphBackdoor) |
| 2025 | Node-level | Backdoor Graph Condensation  | [Paper](https://arxiv.org/pdf/2407.11025) | [Code](https://github.com/JiahaoWuGit/BGC) |
| 2025 | Node-level | Backdoor Attack on Vertical Federated Graph Neural Network Learning  | [Paper](https://arxiv.org/pdf/2410.11290?)| [Code](无) |
| 2025 | Hybrid-level | Adaptive Backdoor Attacks with Reasonable Constraints on Graph Neural Networks  | [Paper](https://arxiv.org/pdf/2503.09049) | [无] |
| 2025 | Link-prediction | Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems  | [Paper](https://arxiv.org/pdf/2506.08401?) | [无]|

> **添加新条目**：在表格最后一行添加 `| 年份 | 层级 | 方法名 | [Paper](链接) | [Code](链接或 (No official code)) |`

---

# Defense Methods

> 表格按：年份 | 防御阶段 | 方法名 | 论文链接 | 代码链接（有则写）

| Year | Stage | Method | Paper | Code |
|------|-------|--------|-------|------|
| 2021 | Testing-time | Backdoor Attacks to Graph Neural Networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3450569.3463560) | [Code](https://github.com/zaixizhang/graphbackdoor) |
| 2022 | Testing-time | 图神经网络后门攻击及防御机制研究 | 中南大学2022陈榕] | No code |
| 2022 | Testing-time | Defending against backdoor attack on graph neural network by explainability | [Paper](https://arxiv.org/pdf/2209.02902) |No code |
| 2023 | Training-time | XGBD: Explanation-guided Graph Backdoor Detection | [Paper](https://arxiv.org/pdf/2308.04406) |  [Code](https://github.com/GuanZihan/GNN_backdoor_detection) |
| 2023 | Training-time | 基于对比学习的图神经网络后门攻击防御方法 | 通信学报 | (Code: TODO) |
| 2024 | Training-time | Securing GNNs: Explanation-based identification of backdoored training graphs | [Paper](T[ODO_SPECTRAL_PAPER_LINK](https://arxiv.org/pdf/2403.18136v1)) | No code|
| 2024 | Training-time | On the Robustness of Graph Reduction Against GNN Backdoor | [Paper](https://arxiv.org/pdf/2407.02431?) |  [Code](https://github.com/GuanZihan/GNN_backdoor_detection) |
| 2024 | Training-time | DMGNN: Detecting and Mitigating Backdoor Attacks in Graph Neural Networks | [Paper](https://arxiv.org/pdf/2410.14105) | No code |
| 2022 | Testing-time | 面向图神经网络后门攻击的触发器恢复和移除方法 | 西电2024辛柳莹] | No code |
| 2024 | Federated | Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses| [Paper](https://dl.acm.org/doi/abs/10.1145/3658644.3690187) |  [Code](https://github.com/Yuxin104/Opt-GDBA)  |
| 2024 | Testing-time | Defense-as-a-service: Black-box shielding against backdoored graph models | [Paper](https://arxiv.org/abs/2410.04916) | No code|
| 2025 | Training-time | Robustness Inspired Graph Backdoor Defense | [Paper](https://arxiv.org/pdf/2406.09836) |  [Code](https://github.com/zzwjames/RIGBD) |
| 2025 | Federated | Energy-based backdoor defense against federated graph learnin | [Paper]([TODO_FEDTGE_PAPER_LINK](https://openreview.net/pdf?id=5Jc7r5aqHJ)) | [Code](https://github.com/ZitongShi/fedTGE) |
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
## Benchmark: GNN Backdoor Results (ASR/CAD, %)

<div style="overflow-x:auto;">

| 任务类型 | 后门攻击方法 | Cora(81) | CiteSeer(87) | PubMed(88) | MUTAG(89) | Mutagenicity(84) | NCI1(86) | PROTEINS(84) |
|---------|-------------|----------|-------------|------------|-----------|-----------------|----------|-------------|
| 节点分类 | General-Backdoor(84) | - | - | - | - | - | - | - |
| 节点分类 | NFTA(93) | - | 83.48/3.39 | - | - | - | - | - |
| 节点分类 | GCBA(98) | 96.2/0.2 | 94.6/0.2 | - | - | - | - | - |
| 节点分类 | UGBA(97) | 96.95/- | - | 92.27/- | - | - | - | - |
| 节点分类 | CBAG(98) | 99.83/2.41 | 96.8/-1.93 | 98.3/1.38 | - | - | - | - |
| 节点分类 | CGBA(98) | 98/5.6 | 99.2/-2.4 | 89.8/-0.1 | - | - | - | - |
| 节点分类 | PecCBA(80) | 89.6/0.27 | 70.34/2.55 | 91.85/3.52 | - | - | - | - |
| 节点分类 | Spectrum-Backdoor(81) | 100/1.2 | 98.2/-0.1 | 100/0.1 | - | - | - | - |
| 节点分类 | BGC(91) | 100/- | 100/- | - | - | - | - | - |
| 节点分类 | BVG(82) | 99.86/- | - | 100/- | - | - | - | - |
| 节点分类 | Multi-target & multi-trigger(43) | 97.75/- | 97.35/- | - | - | - | - | - |
| 节点分类 | MLGB(84) | 98.15/- | 98.51/- | - | - | - | - | - |
| 节点分类 | SBA(85) | 99.21/-0.55 | - | 99.36/0.33 | - | - | - | - |
| 图分类 | Subgraph-Trigger Backdoor Attack(46) | - | - | - | - | - | 100/- | - |
| 图分类 | CLB(87) | - | - | - | - | 87.83/0.16 | - | 98.47/0.88 |
| 图分类 | ESA(48) | - | - | - | - | - | 0.01/- | 0.005/- |
| 图分类 | Motif-backdoor(49) | - | - | - | - | - | 99.72/2.18 | 89.08/- |
| 图分类 | TRAP(50) | - | - | - | - | - | - | 77.78/2.24 |
| 链接预测 | CBD、CBDA(31) | - | - | - | - | - | - | 2.21/- |
| 链接预测 | CBA、CBDA(32) | - | - | - | - | - | - | 4.45/- |
| 链接预测 | Opt-GDBA(30) | - | - | - | - | 85/- | - | 2.74/- |
| 链接预测 | Link-Backdoor(54) | 92.56/1.08 | 98.78/4.97 | 83.25/2.96 | - | - | - | 90/- |
| 链接预测 | BALP(62) | 99.16/0.07 | 99.94/0.39 | - | - | - | - | - |
| 链接预测 | Dyn-backdoor(57) | - | - | - | - | - | - | - |
| 混合型 | GTA(58) | - | - | - | - | - | - | - |
| 混合型 | EBA(57) | 84.22/1.95 | 96.26/1.72 | - | - | 97.69/2.65 | - | - |
| 混合型 | ABARC(56) | 98.16/1.85 | 99.7/1.81 | - | - | - | - | - |

</div>
# Citation

If you use this list or our survey in your work, please cite:

```bibtex
@article{your_survey_2025,
  title={Backdoor Attacks and Defenses in Graph Neural Networks: A Survey},
  author={Your Name and Coauthors},
  year={2025},
  note={arXiv:xxxx.xxxxx or conference info}
}
