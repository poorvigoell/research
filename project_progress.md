# 🚀 Team Project Progress Tracker: ACO-GNN Research

**Project Title:** Pheromone-Guided Graph Neural Networks for Adaptive Fraud Ring Detection in Dynamic Lending Networks  
**Target Output:** B.Tech / M.Tech Major Project Thesis & IEEE / ACM Conference / Journal Paper Submission  
**Directory:** `c:\Users\poorv\OneDrive\Desktop\research\`  
**Current Date:** September 2026  

---

## 👥 Quick Team Onboarding Guide

Welcome to the ACO-GNN research repository! This file tracks the exact progress, completed components, what remains to be built, and how to run experiments.

### Key Files in this Repository:
- [`design_decisions.md`](file:///c:/Users/poorv/OneDrive/Desktop/research/design_decisions.md) — **Must Read:** Explains *why* we chose every architecture component, dataset, metric, and why alternatives were rejected.
- [`research proposal.md`](file:///c:/Users/poorv/OneDrive/Desktop/research/research%20proposal.md) — The formal research proposal, literature survey, and theoretical formulation.
- [`implementation_plan.md`](file:///c:/Users/poorv/OneDrive/Desktop/research/implementation_plan.md) — The technical specification and 5-stage project roadmap.
- [`main.py`](file:///c:/Users/poorv/OneDrive/Desktop/research/main.py) — The CLI experiment runner for all models, baselines, and ablation studies.
- `src/` — Modular source code (ACO engine, models, data generator, training pipelines, metrics, plotting).

---

## 📊 Stage-by-Stage Progress Overview

```
[██████████] Stage 1: Environment & Synthetic Data Pipeline (100% Complete)
[████████░░] Stage 2: Baselines Suite (80% Complete - MLP, GCN, GAT, GraphSAGE, XGBoost, LightGBM Done; Fraud SOTA Stretch)
[█████████░] Stage 3: ACO-GNN Core Architecture (90% Complete - Model, Scatter Softmax, Checkpoint Done)
[██████░░░░] Stage 4: Empirical Evaluation & Ablations (60% - Multi-seed & Visualization Code Ready; Real Data Runs Left)
[██░░░░░░░░] Stage 5: Paper Manuscript Preparation (20% Complete - Figures & Plan Ready; LaTeX Manuscript Left)
```

---

## 🛠 Detailed Milestone Breakdown

### Stage 1: Environment & Data Pipeline
- [x] Python virtual environment setup (`.venv`) with PyTorch, Scikit-learn, SciPy, Matplotlib, Seaborn.
- [x] Pinned `requirements.txt` and repository `.gitignore`.
- [x] Synthetic multi-hop dynamic lending network generator (`src/data/synthetic_generator.py`):
  - [x] Multi-entity heterogeneous projection.
  - [x] Controlled collusive fraud syndicates (3 to 6 hops).
  - [x] Tunable camouflage edge injection (0% to 50%).
  - [x] Chronological application timestamps.
  - [x] Fixed-seed reproducibility across camouflage noise levels.
- [x] Dataset loader interface (`src/data/dataset_loader.py`):
  - [x] Synthetic lending generator loader.
  - [ ] DGraphFin full download and PyG dataset setup (NeurIPS 2022).
  - [ ] YelpChi multi-relational graph download and preprocessing.

### Stage 2: Reproducible Baselines Suite
- [x] Pure PyTorch Baseline Models (`src/models/baselines.py`):
  - [x] Tabular MLP Baseline (node features only, no graph).
  - [x] Standard 2-layer GCN with degree normalization & self-loops.
  - [x] Vectorized GAT with numerically stable scatter softmax.
- [x] Tree-based Baselines (`src/models/tree_baselines.py`):
  - [x] XGBoost (features + degree + PageRank + neighbor fraud ratio + 2-hop reach).
  - [x] LightGBM (same graph-enriched feature set).
- [x] GraphSAGE Baseline (`src/models/baselines.py`):
  - [x] Pure PyTorch GraphSAGE with mean/max aggregation and L2 normalization.
- [ ] Fraud-Specific SOTA Baselines (stretch goal):
  - [ ] CARE-GNN (RL-based neighbor filtering baseline, KDD 2020).
  - [ ] BWGNN (Spectral band-pass GNN baseline, ICDM 2021).
  - [ ] H2-FDetector (Heterophilic fraud GNN, WWW 2022).

### Stage 3: Proposed ACO-GNN Architecture
- [x] Vectorized Pheromone Engine (`src/aco/matrix_ant.py`):
  - [x] Memory-efficient 1D edge pheromone storage.
  - [x] Relation-aware softmax pheromone normalization tau_ij^(r).
  - [x] Closed-loop pheromone deposit based on joint prediction confidence P_i * P_j.
  - [x] Strict label masking (training labels only; zero validation/test leakage).
  - [x] Monotonic evaporation step with boundary clamping [tau_min, tau_max].
- [x] Pheromone-Guided Graph Neural Network (`src/models/aco_gnn.py`):
  - [x] `PheromoneGuidedConv` layer with dual-attention blending.
  - [x] Per-destination-node scatter attention softmax.
  - [x] Learnable sigmoid parameter sigma(lambda) with `fixed_lambda` support for ablations.
  - [x] Deep `ACORingGNN` architecture with LayerNorm, ReLU, and residual skip connections.
  - [x] Multi-layer classification MLP head.
- [x] Training & Optimization Pipeline (`src/training/train_eval.py`):
  - [x] Focal Loss criterion (alpha=0.75, gamma=2.0) for severe class imbalance.
  - [x] Early stopping on validation minority Illicit-F1.
  - [x] Dynamic threshold calibration via validation Precision-Recall curve.
  - [x] Synchronous checkpointing of model weights and pheromone values.

### Stage 4: Empirical Experiments & Results
- [x] Main Model Comparison (15% Camouflage Noise):
  - [x] **ACO-GNN achieves 0.9412 Illicit-F1 vs GAT 0.3125 (+0.6287 lead).**
  - [x] Saved to `experiments/results/baselines_results.json` & `aco_gnn_results.json`.
- [x] Camouflage Attack Robustness Stress Test (0% to 50% noise):
  - [x] GAT degrades from 0.6415 down to 0.1304.
  - [x] ACO-GNN retains robust performance (0.5854 to 0.8519).
  - [x] Saved to `experiments/results/camouflage_experiment.json`.
  - [x] Publication-quality 300 DPI plot: `paper/figures/camouflage_robustness.pdf` and `.png`.
- [x] Component Ablation Study (20% Camouflage Noise):
  - [x] Vanilla GAT: 0.4444 F1
  - [x] Pure ACO (w/o attention): 0.8148 F1
  - [x] No Evaporation (rho=0): 0.8679 F1
  - [x] Full Proposed ACO-GNN: 0.8519 F1 (highest PR-AUC 0.9179, AUC-ROC 0.9925).
  - [x] Publication-quality 300 DPI plot: `paper/figures/ablation_study.pdf` and `.png`.
- [x] Multi-seed variance runner implemented (`main.py --task multi_seed`, 5 seeds, Mean +/- Std). Awaiting execution.
- [x] t-SNE visualization and embedding extraction code implemented (`src/utils/plotting.py`). Awaiting execution.
- [x] Pheromone trail heatmap code implemented (`src/utils/plotting.py`). Awaiting execution.
- [ ] Execute multi-seed runs on synthetic, DGraphFin, and YelpChi datasets.
- [ ] Generate t-SNE and pheromone heatmap figures for paper.

### Stage 5: Paper Manuscript & Deliverables
- [x] Research proposal synchronized with modern 2022-2025 literature.
- [x] Mathematical framework fully documented in LaTeX notation.
- [ ] Full LaTeX manuscript (`paper/aco_gnn_manuscript.tex` in IEEE/ACM double-column format).
- [ ] BibTeX reference library (`paper/references.bib`).
- [ ] Final project defense presentation slides.

---

## 💻 How to Run the Code Locally

Open PowerShell in `c:\Users\poorv\OneDrive\Desktop\research\`:

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Test the Synthetic Data Pipeline
python main.py --task test_data

# 3. Train the Proposed ACO-GNN Model
python main.py --task train_aco_gnn --epochs 25

# 4. Run Baseline Models (MLP, GCN, GAT)
python main.py --task run_baselines --epochs 25

# 5. Run Tree Baselines (XGBoost + LightGBM with graph features)
python main.py --task run_tree_baselines --dataset synthetic

# 6. Run Camouflage Stress Test (0% to 50% Noise)
python main.py --task camouflage_experiment --epochs 25

# 7. Run Component Ablation Study
python main.py --task ablation_study --epochs 25

# 8. Multi-Seed Benchmark (5 seeds, all models, Mean +/- Std)
python main.py --task multi_seed --epochs 25 --dataset synthetic

# 9. Run on Real-World DGraphFin (requires torch-geometric)
python main.py --task multi_seed --epochs 30 --dataset dgraphfin

# 10. Run on YelpChi Cross-Domain Benchmark
python main.py --task multi_seed --epochs 30 --dataset yelpchi
```

---

## 🎯 Next Priority Work Items
1. **Execute multi-seed runs** on synthetic dataset and collect Mean +/- Std tables.
2. **Download and run DGraphFin** experiments for real-world industrial-scale results.
3. **Generate t-SNE and pheromone heatmap** figures after training ACO-GNN.
4. **Draft LaTeX manuscript** in IEEE double-column format with all results and figures.
