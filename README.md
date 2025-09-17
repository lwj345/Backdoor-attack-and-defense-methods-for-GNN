# Backdoor-attack-and-defense-methods-for-GNN
A list of papers, codes, and resources on Graph Neural Network (GNN) backdoor attacks and defenses.

---

## Contents
- [Attack Methods](#attack-methods)
- [Defense Methods](#defense-methods)
- [Structure](#structure)
- [Citation](#citation)
- [Contributing](#contributing)

---

# Attack Methods

> 表格按：年份 | 任务层级 | 方法名 | 论文链接 | 代码链接（有则写，没有写 No official code）

| Year | Level | Method | Paper | Code |
|------|-------|--------|-------|------|
| 2021 | Graph-level | Backdoor Attacks to Graph Neural Networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3450569.3463560) | [Code](https://github.com/zaixizhang/graphbackdoor) |
| 2021 | Graph-level | Backdoor attack of graph neural networks based on subgraph trigger | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-92638-0_17) | (Code:  `No official code`) |
| 2021 | Hybrid-level | Graph backdoor | [Paper](https://www.usenix.org/system/files/sec21-xi.pdf) | [Code](https://github.com/zhaohan-xi/GraphBackdoor) |
| 2021 | Hybrid-level  | Explainability-based backdoor attacks against graph neural networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3468218.3469046) | (Code:  `No official code`) |
| 2022 | Node-level  | A general backdoor attack to graph neural networks based on explanation method | [Paper](https://ieeexplore.ieee.org/abstract/document/10063438) | (Code: `No official code`) |
| 2022 | Graph-level | Poster: Clean-label Backdoor Attack on Graph Neural Networks| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002000) | (Code:  `No official code`)  |
| 2022 | Graph-level | Transferable Graph Backdoor Attack| [Paper](https://arxiv.org/pdf/2207.00425) | (Code:  `No official code`)  |
| 2022 | Graph-level | More is better (mostly): On the backdoor attacks in federated graph neural networks| [Paper](https://dl.acm.org/doi/abs/10.1145/3564625.3567999) |  [Code](https://github.com/xujing1994/bkd_fedgnn)  |
| 2023 | Node-level  | Feature-Based Graph Backdoor Attack in the Node Classification Task | [Paper](https://onlinelibrary.wiley.com/doi/pdf/10.1155/2023/5418398) | (Code: `No official code`) |
| 2023 | Node-level  | Graph contrastive backdoor attacks | [Paper](https://proceedings.mlr.press/v202/zhang23e/zhang23e.pdf) | (Code:  `No official code`) |
| 2023 | Node-level  | Unnoticeable Backdoor Atacks on Graph Neural Networks | [Paper](https://dl.acm.org/doi/10.1145/3543507.3583392) | [Code](https://github.com/ventr1c/UGBA) |
| 2023 | Node-level  | A Clean-graph Backdoor Attack against Graph Convolutional Networks with Poisoned Label Only | [Paper](https://arxiv.org/pdf/2404.12704) | (Code:  `No official code`)  |
| 2023 | Node-level  | PerCBA: Persistent Clean-label Backdoor Attacks on Semi-Supervised Graph Node Classification | [Paper](https://ceur-ws.org/Vol-3505/paper_4.pdf) | (Code:  `No official code`)  |
| 2023 | Node-level  | Effective Backdoor Attack on Graph Neural Networks in Spectral Domain | [Paper](https://ieeexplore.ieee.org/abstract/document/10318195) | (Code:  `No official code`)  |
| 2023 | Node-level  | Poster: Multi-target & Multi-trigger Backdoor Attacks on Graph Neural Networks | [Paper](https://repository.ubn.ru.nl/handle/2066/299150) | (Code:  `No official code`)  |
| 2023 | Graph-level | Motif-backdoor: Rethinking the backdoor attack on graph neural networks via motif| [Paper](https://arxiv.org/pdf/2210.13710) |  [Code](https://github.com/Seaocn/Motif-Backdoor)  |
| 2023 | Link-prediction | Link-backdoor: Backdoor attack on link prediction via node injection| [Paper](https://ieeexplore.ieee.org/abstract/document/10087329) |  [Code](https://github.com/Seaocn/Link-Backdoor)  |
| 2023 | Link-prediction | Dyn-Backdoor: Backdoor Attack on Dynamic Link Prediction| [Paper](https://arxiv.org/pdf/2110.03875) |  (Code:  `No official code`)  |
| 2023 | Hybrid-level  | Bkd-fedgnn: A benchmark for classification backdoor attacks on federated graph neural network | [Paper](https://arxiv.org/pdf/2306.10351) | [Code](https://github.com/usail-hkust/BkdFedGCN) |
| 2024 | Node-level  | Multi-target label backdoor attacks on graph neural networks | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002000) | (Code:  `No official code`)  |
| 2024 | Node-level  | A clean-label graph backdoor attack method in node classification task | [Paper](https://arxiv.org/pdf/2401.00163) | (Code:  `No official code`)  |
| 2024 | Graph-level | Explanatory subgraph attacks against Graph Neural Networks| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608024000030) | (Code:  `No official code`)  |
| 2024 | Graph-level | Unveiling the threat: Investigating distributed and centralized backdoor attacks in federated graph neural network| [Paper](https://dl.acm.org/doi/full/10.1145/3633206) | (Code:  `No official code`)  |
| 2024 | Graph-level | Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses| [Paper](https://dl.acm.org/doi/abs/10.1145/3658644.3690187) |  [Code](https://github.com/Yuxin104/Opt-GDBA)  |
| 2024 | Link-prediction | A backdoor attack against link prediction tasks with graph neural networks| [Paper](https://arxiv.org/pdf/2401.02663) |  (Code:  `No official code`)  |
| 2024 | Hybrid-level | 自适应图神经网络后门攻击方法研究 | [Paper]西电硕士论文 | [Code](https://github.com/zhaohan-xi/GraphBackdoor) |



> **添加新条目**：在表格最后一行添加 `| 年份 | 层级 | 方法名 | [Paper](链接) | [Code](链接或 (No official code)) |`

---

# Defense Methods

> 表格按：年份 | 防御阶段 | 方法名 | 论文链接 | 代码链接（有则写）

| Year | Stage | Method | Paper | Code |
|------|-------|--------|-------|------|
| 2021 | Testing-time | Backdoor Attacks to Graph Neural Networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3450569.3463560) | [Code](https://github.com/zaixizhang/graphbackdoor) |
| 2022 | Testing-time | 图神经网络后门攻击及防御机制研究 | [Paper中南大学2022陈榕] | (Code:  `No official code`) |
| 2022 | Testing-time | Defending against backdoor attack on graph neural network by explainability | [Paper](https://arxiv.org/pdf/2209.02902) | (Code:  `No official code`) |
| 2023 | Training-time | XGBD: Explanation-guided Graph Backdoor Detection | [Paper](https://arxiv.org/pdf/2308.04406) |  [Code](https://github.com/GuanZihan/GNN_backdoor_detection) |
| 2023 | Training-time | 基于对比学习的图神经网络后门攻击防御方法 | [Paper中文期刊] | (Code: TODO) |
| 2024 | Training-time | Securing GNNs: Explanation-based identification of backdoored training graphs | [Paper](T[ODO_SPECTRAL_PAPER_LINK](https://arxiv.org/pdf/2403.18136v1)) | (Code: TODO) |
| 2024 | Training-time | On the Robustness of Graph Reduction Against GNN Backdoor | [Paper](https://arxiv.org/pdf/2407.02431?) |  [Code](https://github.com/GuanZihan/GNN_backdoor_detection) |
| 2024 | Training-time | DMGNN: Detecting and Mitigating Backdoor Attacks in Graph Neural Networks | [Paper](https://arxiv.org/pdf/2410.14105) |  (Code:  `No official code`)  |
| 2022 | Testing-time | 面向图神经网络后门攻击的触发器恢复和移除方法 | [Paper西电2024辛柳莹] | (Code:  `No official code`) |
| 2024 | Federated | Distributed Backdoor Attacks on Federated Graph Learning and Certified Defenses| [Paper](https://dl.acm.org/doi/abs/10.1145/3658644.3690187) |  [Code](https://github.com/Yuxin104/Opt-GDBA)  |
| 2024 | Testing-time | Defense-as-a-service: Black-box shielding against backdoored graph models | [Paper](https://arxiv.org/abs/2410.04916) | (Code:  `No official code`) |
| 2025 | Training-time | Robustness Inspired Graph Backdoor Defense | [Paper](https://arxiv.org/pdf/2406.09836) |  [Code](https://github.com/zzwjames/RIGBD) |
| 2025 | Federated | Energy-based backdoor defense against federated graph learnin | [Paper]([TODO_FEDTGE_PAPER_LINK](https://openreview.net/pdf?id=5Jc7r5aqHJ)) | [Code](https://github.com/ZitongShi/fedTGE) |

---

# Structure

- `README.md` — this file (paper list + links)  
- `papers/` — optional: store PDFs or BibTeX files (upload via "Add file → Upload files")  
- `code/` — optional: if you include any local example code (or link out to external repos)  
- `scripts/` — optional: download / reproduce scripts

---

# Citation

If you use this list or our survey in your work, please cite:

```bibtex
@article{your_survey_2025,
  title={Backdoor Attacks and Defenses in Graph Neural Networks: A Survey},
  author={Your Name and Coauthors},
  year={2025},
  note={arXiv:xxxx.xxxxx or conference info}
}
