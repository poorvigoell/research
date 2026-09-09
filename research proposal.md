# 🎓 MAJOR PROJECT RESEARCH PROPOSAL

**Project Title:** Pheromone-Guided Graph Neural Networks for Adaptive Fraud Ring Detection in Dynamic Lending Networks  
**Domain:** Graph Machine Learning, Swarm Intelligence, Financial Technology Security  
**Student Name(s):** [Your Name] & [Teammate Name]  
**Target Output:** B.Tech / M.Tech Major Project Thesis & Conference / Journal Paper Submission (IEEE / ACM)  

---

## 1. ABSTRACT & EXECUTIVE SUMMARY

Financial fraud in digital lending networks has evolved from isolated individual defaults to organized, multi-hop fraud syndicates. Traditional rule-based engines and static machine learning scorecards fail to capture complex structural relationships across heterogeneous entities (e.g., shared devices, bank accounts, guarantors, and payment channels). While Graph Neural Networks (GNNs) have emerged as a powerful tool for graph-structured fraud detection, existing GNN architectures suffer from **neighborhood over-smoothing**, vulnerability to **graph camouflage attacks** (where fraudsters connect to legitimate entities to evade detection), and **high computational latency** on time-evolving dynamic graphs.

This project proposes **ACO-GNN**, a novel hybrid framework that integrates **Ant Colony Optimization (ACO) stigmergic principles directly into the GNN message-passing mechanism**. By deploying stochastic ant agents on dynamic heterogeneous knowledge graphs, the system continuously updates artificial pheromone trails ($\tau_{ij}$) along suspicious structural paths. These pheromone intensities dynamically scale the attention weights of graph aggregation layers, allowing the network to adaptively filter out camouflaged noise while zeroing in on emerging multi-hop fraud rings. The framework will be evaluated on large-scale real-world industrial and benchmark datasets (DGraphFin, YelpChi, Amazon, and Synthetic Lending Networks) against classic and modern state-of-the-art baselines (CARE-GNN, BWGNN, H2-FDetector, GAGA, SEC-GFD).

---

## 2. PROBLEM STATEMENT & RESEARCH GAP

### 2.1 The Problem
In modern credit lending networks, fraud rings intentionally obscure their footprints by:
1. **Multi-hop dispersion:** Spreading connections across 4–6 degrees of separation (e.g., Applicant A $\rightarrow$ Device $\rightarrow$ Applicant B $\rightarrow$ Guarantor $\rightarrow$ Phone $\rightarrow$ Applicant D).
2. **Behavioral & Structural Camouflage:** Interlinking with high-credit, legitimate entities and maintaining normal repayment behavior for initial cycles ("3-and-ghost" pattern) before executing coordinated defaults.
3. **Concept Drift & Mutation:** Altering network topologies dynamically across geographic regions once a specific pattern is flagged.

### 2.2 The Research Gap
A comprehensive survey of 80+ domain papers reveals a critical academic gap:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ACADEMIC LANDSCAPE                              │
│                                                                             │
│  Existing Focus A:              Existing Focus B:                           │
│  Graph Neural Networks (GNNs)   ACO + GNN Hybrids                           │
│  • Effective at node topology   • Exist for combinatorial optimization      │
│  • FAILS under graph camouflage   (DeepACO NeurIPS 2023, GHNN-ACO 2023,     │
│  • Heavy computational lag        GFACS ICLR 2024)                          │
│    on time-evolving graphs      • NOT applied to heterogeneous dynamic      │
│                                   graphs for node classification            │
│                                      │                                      │
│                                      ▼                                      │
│                          🔴 WIDE-OPEN RESEARCH GAP                          │
│  No existing work integrates stigmergic pheromone dynamics into GNN         │
│  message-passing for graph anomaly detection or financial fraud, where      │
│  the goal is node classification on heterogeneous dynamic graphs.           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. PROPOSED METHODOLOGY: THE ACO-GNN FRAMEWORK

The proposed **ACO-GNN** architecture consists of three tightly coupled components:

```
Raw Transaction / ──► Heterogeneous ──► ACO Pheromone ──► Pheromone-Guided ──► Classification
Application Data      Knowledge Graph    Agent Traversal    GNN Message Pass   & Risk Verdict
```

### 3.1 Heterogeneous Graph Construction
The dynamic lending network is modeled as a heterogeneous graph $G = (V, E, T, R)$ where:
* **Nodes ($V$):** Applicants, Devices (IMEI/IP), Bank Accounts, Dealers, Guarantors, Locations.
* **Edges ($E$):** `SHARES_DEVICE`, `DISBURSED_TO`, `SOURCED_BY`, `GUARANTEES`, `LOCATED_NEAR`.
* **Timestamps ($T$):** Temporal markers for every relation.
* **Relation Types ($R$):** DGraphFin provides 11 distinct emergency contact relationship types; YelpChi provides 3 meta-path relations (R-U-R, R-T-R, R-S-R). Relation-specific pheromone matrices $\tau_{ij}^{(r)}$ are maintained per edge type.

### 3.2 Pheromone-Guided Stigmergic Traversal (ACO Layer)
Virtual ant agents crawl the graph according to a transition probability $P_{ij}^k$ influenced by both structural topology and node risk heuristics $\eta_{ij}$:

$$P_{ij}^k(t) = \frac{[\tau_{ij}(t)]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in N_i} [\tau_{il}(t)]^\alpha \cdot [\eta_{il}]^\beta}$$

* **Pheromone Reinforcement:** When confirmed defaults or high-risk anomalies occur, edge pheromone concentrations $\tau_{ij}$ are exponentially reinforced ($\tau_{ij} \leftarrow \tau_{ij} + \Delta\tau$).
* **Pheromone Evaporation:** Benign/transient paths decay over time ($\tau_{ij} \leftarrow (1-\rho)\tau_{ij}$), automatically dissolving false alarms (e.g., shared public Wi-Fi).

### 3.3 Pheromone-Weighted GNN Aggregation Layer
Instead of standard uniform or attention-based aggregation (GCN/GAT), the node hidden representation $h_i^{(l+1)}$ at layer $l+1$ is computed by scaling neighbor messages using normalized pheromone weights $\hat{\tau}_{ij}$:

$$h_i^{(l+1)} = \sigma \left( W_{\text{root}} \cdot h_i^{(l)} + \sum_{r \in \mathcal{R}} W_r \sum_{j \in N_r(i)} \left( \lambda \cdot \alpha_{ij}^{(r)} + (1 - \lambda) \cdot \hat{\tau}_{ij}^{(r)} \right) h_j^{(l)} \right)$$

where $\alpha_{ij}^{(r)}$ is GAT self-attention, $\hat{\tau}_{ij}^{(r)} = \frac{\exp(\tau_{ij}^{(r)} / T)}{\sum_{k \in N_r(i)} \exp(\tau_{ik}^{(r)} / T)}$ is the relation-aware pheromone prior with temperature $T$, and $\lambda \in [0, 1]$ is an adaptive blending coefficient.

**Key Advantage:** The pheromone trail provides **persistent structural memory** across training iterations—unlike GAT attention (recomputed per forward pass) or CARE-GNN's RL filtering (per-node policy), pheromone trails accumulate multi-hop evidence over time and automatically decay false signals via evaporation, enabling detection of 4–6 hop fraud rings that single-pass attention mechanisms miss.

---

## 4. EXPERIMENTAL SETUP & EVALUATION PLAN

### 4.1 Datasets

| Dataset | Type | Entities / Nodes | Edges / Transactions | Task |
| :--- | :--- | :--- | :--- | :--- |
| **DGraphFin** | Real-world Financial Lending Network (NeurIPS 2022) | 3,700,550 | 4,300,999 (11 edge types) | Semi-supervised Node Classification (Fraud vs. Normal) |
| **YelpChi / Amazon** | Benchmark Fraud Graph | ~45,000 / ~12,000 | ~3,846,979 / ~4,398,392 | Cross-Domain Camouflaged Fraud Node Detection |
| **Synthetic Lending Graph** | Controlled Heterogeneous Multi-hop Network | Configurable (10k–1M) | Multi-entity relational | Multi-hop Ring Discovery & Camouflage Stress Test |

### 4.2 Baseline Models for Comparison
We will benchmark **ACO-GNN** against standard, fraud-specific, and state-of-the-art architectures:
1. **Classic ML:** XGBoost, LightGBM (Tabular baseline)
2. **Standard GNNs:** GCN, GraphSAGE, GAT
3. **Fraud-Specific GNNs (2020–2021):** CARE-GNN, PC-GNN, BWGNN, FRAUDRE
4. **Modern Fraud GNNs (2022–2025):** H2-FDetector (WWW 2022), GAGA (WWW 2023), SEC-GFD (AAAI 2024), DGA-GNN (AAAI 2024)

### 4.3 Evaluation Metrics
* **Classification Performance:** Minority Illicit-F1, F1-Macro, Area Under ROC Curve (AUC-ROC), Precision-Recall AUC (PR-AUC). Primary optimization target: **Illicit-F1** and **PR-AUC** (to account for severe class imbalance).
* **Efficiency & Latency:** Inference time per graph batch (ms), memory footprint, convergence rate.
* **Robustness:** Detection accuracy under varying levels of graph camouflage noise (10% to 50% camouflage edges).

---

## 5. PROJECT MILESTONES & TIMELINE

```
Phase 1: Literature Review & Setup (Weeks 1–4)
██████████████
• Complete literature survey on GNNs & Swarm Intelligence
• Set up PyTorch Geometric (PyG) and DGL environment
• Benchmark baseline GCN/GAT/GraphSAGE models on DGraphFin & YelpChi

Phase 2: Framework Development (Weeks 5–10)
              ████████████████████
              • Implement vectorized ACO graph agent simulation module
              • Design custom Pheromone-Guided Aggregation GNN Layer in PyTorch
              • Build dynamic pheromone update & evaporation algorithms

Phase 3: Experimentation & Tuning (Weeks 11–16)
                                  ████████████████████
                                  • Run comparative experiments against CARE-GNN, BWGNN, H2-FDetector, and GAGA
                                  • Evaluate camouflage robustness & latency metrics across DGraphFin & YelpChi/Amazon
                                  • Ablation study (ACO only vs GNN only vs ACO-GNN variants)

Phase 4: Paper Writing & Final Thesis (Weeks 17–20)
                                                      ██████████████
                                                      • Compile experimental results & visualizations
                                                      • Draft research paper following IEEE conference template
                                                      • Finalize project report & demonstration
```

---

## 6. EXPECTED OUTCOMES & DELIVERABLES

1. **A Novel Open-Source Framework:** A fully functional PyTorch Geometric implementation of ACO-GNN for dynamic fraud detection.
2. **Empirical Benchmarks:** Quantitative proof demonstrating that ACO-GNN outperforms existing GNNs under camouflage noise and multi-hop settings.
3. **Publication:** A high-quality research paper targetable for Scopus / Web of Science indexed IEEE/ACM conferences (e.g., IEEE BigData, ICDM, or IEEE Access).
4. **Major Project Thesis:** Complete documentation adhering to university guidelines.

---

## 7. HARDWARE & SOFTWARE REQUIREMENTS

* **Programming Language:** Python 3.10+
* **Core Libraries:** PyTorch, PyTorch Geometric (PyG), NetworkX, Deep Graph Library (DGL), NumPy, Scikit-learn
* **Compute Infrastructure:** Google Colab Pro / University GPU Workstation (NVIDIA RTX 3080/4090 or T4/V100)
* **Visualization:** D3.js, Matplotlib, Gephi

---

## 8. REFERENCES (KEY PAPERS TO CITE)

1. Dou, Y., et al. (2020). *Enhancing Graph Neural Networks for Fraud Detection via Neighbor Filtering (CARE-GNN)*. **ACM SIGKDD**.
2. Liu, Z., et al. (2021). *Pick and Choose: A GNN-based Fraud Detection Framework on Graphs (PC-GNN)*. **WWW '21**.
3. Wang, U., et al. (2021). *Anti-Fraud Graph Neural Networks via Bandpass Filtering (BWGNN)*. **IEEE ICDM**.
4. Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. **MIT Press**.
5. Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*. **arXiv:1908.02591**.
6. Huang, X., et al. (2022). *DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection*. **NeurIPS 2022 Datasets & Benchmarks Track**.
7. Shi, F., et al. (2022). *H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections*. **WWW '22**.
8. Wang, Y., et al. (2023). *Label Information Enhanced Fraud Detection against Low Homophily in Graphs (GAGA)*. **WWW '23**.
9. Ye, H., et al. (2023). *DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization*. **NeurIPS 2023**.
10. Zhang, G., et al. (2021). *FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance*. **IEEE ICDM 2021 (Best Student Paper)**.
11. Xu, F., et al. (2024). *Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum (SEC-GFD)*. **AAAI 2024**.
12. Duan, M., et al. (2024). *DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection*. **AAAI 2024**.

---

## 9. APPENDIX: COMPLETE CONTEXT & LITERATURE SURVEY SUMMARY

*(This section preserves all technical research findings, literature survey results, and implementation specs for seamless reference across project iterations.)*

### 9.1 Literature Survey Key Findings (81 Papers Surveyed)
* **GNNs for Fraud Detection (13 papers):** Mature area (CARE-GNN, PC-GNN, BWGNN, HGT). Effective at local structural anomaly detection, but vulnerable to camouflage nodes (edges added to benign entities).
* **Swarm Intelligence for Fraud (13 papers):** Moderate maturity. Almost exclusively used as offline hyperparameter tuners or feature selectors (e.g., PSO-XGBoost).
* **Temporal Graph Networks (10 papers):** Dynamic graphs tracking time-evolving nodes (TGN, DySAT).
* **The ACO + GNN Intersection (8–10 papers globally):** ACO+GNN hybrids exist for **combinatorial optimization** (DeepACO, NeurIPS 2023; GFACS, ICLR 2024; NeuFACO, 2025) and task allocation (GHNN-ACO, 2023). However, **NO existing work integrates ACO stigmergy into GNN message-passing for graph anomaly detection or financial fraud**. Existing fraud-specific GNNs rely exclusively on RL-based neighbor filtering (CARE-GNN), spectral filters (BWGNN, SEC-GFD), or attention/transformers (GAGA). The application of persistent pheromone-based structural memory to fraud ring discovery remains a **distinct, unexplored research gap**.

### 9.2 Key Benchmark Papers to Cite in Proposal / Paper
1. **DGraphFin Dataset:** Huang et al. *"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection"*, NeurIPS 2022.
2. **H2-FDetector:** Shi et al. *"H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections"*, WWW 2022.
3. **GAGA:** Wang et al. *"Label Information Enhanced Fraud Detection against Low Homophily in Graphs"*, WWW 2023.
4. **CARE-GNN:** Dou et al. *"Enhancing Graph Neural Networks for Fraud Detection via Neighbor Filtering"*, KDD 2020.
5. **BWGNN:** Wang et al. *"Anti-Fraud Graph Neural Networks via Bandpass Filtering"*, ICDM 2021.
6. **DeepACO:** Ye et al. *"DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization"*, NeurIPS 2023.

### 9.3 Core Mathematical Innovations
* **ACO Transition Probability:**
  $$P_{ij}^k(t) = \frac{[\tau_{ij}(t)]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in N_i} [\tau_{il}(t)]^\alpha \cdot [\eta_{il}]^\beta}$$
* **Relation-Aware Pheromone Prior:**
  $$\hat{\tau}_{ij}^{(r)} = \frac{\exp(\tau_{ij}^{(r)} / T)}{\sum_{k \in N_r(i)} \exp(\tau_{ik}^{(r)} / T)}$$
* **Pheromone-Guided Dual-Attention Message Passing Layer:**
  $$h_i^{(l+1)} = \sigma \left( W_{\text{root}} \cdot h_i^{(l)} + \sum_{r \in \mathcal{R}} W_r \sum_{j \in N_r(i)} \left( \lambda \cdot \alpha_{ij}^{(r)} + (1 - \lambda) \cdot \hat{\tau}_{ij}^{(r)} \right) h_j^{(l)} \right)$$

---

**Submitted by:**  
Student Name: _______________________ Roll No: ____________ Signature: ___________  
Teammate Name: ____________________ Roll No: ____________ Signature: ___________  

**Approved by Project Supervisor:**  
Supervisor Name: ____________________ Designation: ___________ Signature: ___________  

