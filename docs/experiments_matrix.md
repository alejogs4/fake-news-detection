# 🧪 Test Matrix: 10 Comparative Fake News Detection Experiments

This document describes the 10 rigorous experiments designed for our Gossipcop dataset using BERT feature embeddings. These experiments specifically compare three architectural paradigms: Text-Based, GNN-Based (tree structures), and HGFND (Hypergraph Neural Networks).

---

## 📊 The 10-Experiment Comparative Matrix

| Experiment ID | Experiment Name   | Architecture Type | GNN backbone | Config parameters (`model`)                                                | Rationale & Trade-offs                                                                                                                |
| :--------------| :------------------| :------------------| :-------------| :---------------------------------------------------------------------------| :--------------------------------------------------------------------------------------------------------------------------------------|
| **1**         | `Text-Only`       | Text-Based        | None         | `use_gnn: false`, `use_text: true`, `use_hgfnd: false`                     | **Content Baseline:** Evaluates if linguistic attributes alone can identify fake news, representing purely *endogenous* preferences.  |
| **2**         | `GNN-Only (GCN)`  | Graph-Based       | GCN          | `use_gnn: true`, `use_text: false`, `use_hgfnd: false`, `gnn_type: "GCN"`  | **Propagation Baseline (Symmetric):** Evaluates retweet diffusion cascades alone under a GCN backbone, assuming equal contribution.   |
| **3**         | `GNN-Only (SAGE)` | Graph-Based       | GraphSAGE    | `use_gnn: true`, `use_text: false`, `use_hgfnd: false`, `gnn_type: "SAGE"` | **Propagation Baseline (Inductive):** Evaluates retweet cascades under max-pooled neighborhood aggregation (highly size-robust).      |
| **4**         | `GNN-Only (GAT)`  | Graph-Based       | GAT          | `use_gnn: true`, `use_text: false`, `use_hgfnd: false`, `gnn_type: "GAT"`  | **Propagation Baseline (Attention):** Evaluates tree cascades using trainable attention weights to model user retweeter dependencies. |
| **5**         | `GNN+Text (GCN)`  | Hybrid Fusion     | GCN          | `use_gnn: true`, `use_text: true`, `use_hgfnd: false`, `gnn_type: "GCN"`   | **Early Fusion GCN:** Concatenates GCN-cascade features and news text embeddings directly before classification.                      |
| **6**         | `GNN+Text (SAGE)` | Hybrid Fusion     | GraphSAGE    | `use_gnn: true`, `use_text: true`, `use_hgfnd: false`, `gnn_type: "SAGE"`  | **Early Fusion SAGE:** Concatenates SAGE-cascade features and news text embeddings directly.                                          |
| **7**         | `GNN+Text (GAT)`  | Hybrid Fusion     | GAT          | `use_gnn: true`, `use_text: true`, `use_hgfnd: false`, `gnn_type: "GAT"`   | **Early Fusion GAT:** Concatenates GAT-cascade features and news text embeddings directly.                                            |
| **8**         | `HGFND (GCN)`     | Hypergraph        | GCN          | `use_gnn: true`, `use_text: true`, `use_hgfnd: true`, `gnn_type: "GCN"`    | **Transductive HGFND (GCN):** Models high-order echo chambers transductively with GCN as the GNN propagation cascade encoder.         |
| **9**         | `HGFND (SAGE)`    | Hypergraph        | GraphSAGE    | `use_gnn: true`, `use_text: true`, `use_hgfnd: true`, `gnn_type: "SAGE"`   | **Transductive HGFND (SAGE):** Models high-order echo chambers with GraphSAGE as the GNN cascade encoder. *Paper standard config.*    |
| **10**        | `HGFND (GAT)`     | Hypergraph        | GAT          | `use_gnn: true`, `use_text: true`, `use_hgfnd: true`, `gnn_type: "GAT"`    | **Transductive HGFND (GAT):** Models high-order echo chambers using attention weights on tree-level cascades + dual-level attention.  |

---

## 🛠️ Configuration and Parameter Tuning

All models in the test matrix share standard hyperparameters to ensure a clean, scientific, and unbiased comparison:
*   **Dataset:** `gossipcop` (5,464 news graphs, split 20% train, 10% validation, 70% test).
*   **Input Features:** `bert` (768-dimensional node embeddings from user historical tweets).
*   **Hidden Channels:** `128` (dimension $d$ for all encoders, GNN modules, and attention context projections).
*   **Learning Rate:** `0.001` (optimized Adam optimizer with standard weight decay of `0.01`).
*   **Epochs:** `30` (ideal epoch length for fast local/Colab comparisons to ensure model convergence and prevent overfitting).
