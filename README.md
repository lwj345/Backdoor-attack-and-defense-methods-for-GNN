# Backdoor-attack-and-defense-methods-for-GNN
A list of papers, codes, and resources on Graph Neural Network (GNN) backdoor attacks and defenses.
# Backdoor-attack-and-defense-methods-for-GNN

A curated list of **Graph Neural Network (GNN) backdoor attack and defense** papers and code, organized by task and year.  
This repository accompanies the survey: *"Backdoor Attacks and Defenses in Graph Neural Networks"* (2025).

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
| 2021 | Graph-level | GTA: Graph Trojaning Attack | [Paper](https://www.usenix.org/conference/usenixsecurity21/presentation/xiang) | [Code](https://github.com/PurduePAML/Graph-Trojaning-Attack) |
| 2022 | Graph-level | MLGB | [Paper](TODO_MLGB_PAPER_LINK) | (Code: TODO or `No official code`) |
| 2022 | Node-level  | PoisonedGNN | [Paper](TODO_PoisonedGNN_PAPER_LINK) | (No official code) |
| 2023 | Node-level  | TRAP | [Paper](TODO_TRAP_PAPER_LINK) | [Code](TODO_TRAP_CODE_LINK) |
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
