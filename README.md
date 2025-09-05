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
| 2021 | Graph-level | Backdoor attack of graph neural networks based on subgraph trigger | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-92638-0_17) | (Code: TODO or `No official code`) |
| 2021 | Hybrid-level | Graph backdoor | [Paper](https://www.usenix.org/system/files/sec21-xi.pdf) | [Code](https://github.com/zhaohan-xi/GraphBackdoor) |
| 2021 | Hybrid-level  | Explainability-based backdoor attacks against graph neural networks | [Paper](https://dl.acm.org/doi/pdf/10.1145/3468218.3469046) | (Code: TODO or `No official code`) |
| 2022 | Node-level  | A general backdoor attack to graph neural networks based on explanation method | [Paper](https://ieeexplore.ieee.org/abstract/document/10063438) | (Code: TODO or `No official code`) |
| 2023 | Link-prediction | Transferable Graph Backdoor Attack | [Paper](TODO_TRANSFERABLE_PAPER_LINK) | (Code: TODO) |

> **添加新条目**：在表格最后一行添加 `| 年份 | 层级 | 方法名 | [Paper](链接) | [Code](链接或 (No official code)) |`

---

# Defense Methods

> 表格按：年份 | 防御阶段 | 方法名 | 论文链接 | 代码链接（有则写）

| Year | Stage | Method | Paper | Code |
|------|-------|--------|-------|------|
| 2019 | Training-time (adapt) | Spectral Signatures (adapted for GNNs) | [Paper](TODO_SPECTRAL_PAPER_LINK) | (Code: TODO) |
| 2023 | Training-time | XGBD: Explanation-guided Graph Backdoor Detection | [Paper](https://arxiv.org/abs/2303.XXXX) | (Code: TODO) |
| 2022 | Testing-time | STRIP for Graphs | [Paper](TODO_STRIP_PAPER_LINK) | (Code: TODO) |
| 2025 | Federated | FedTGE: Topology-energy based defense | [Paper](TODO_FEDTGE_PAPER_LINK) | (Code: TODO) |

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
