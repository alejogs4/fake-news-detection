# Learning Guide: User-Preference-aware Fake News Detection (UPFD)

This project implements the UPFD framework as described in the paper: **"User-Preference-aware Fake News Detection" (SIGIR 2021)**.

## Core Concepts

### 1. The Limitation of Traditional Methods
Most fake news detection systems look at the news content (text) or how it spreads. However, they often ignore the **users** who are spreading it.

### 2. Confirmation Bias Theory
Social science tells us that users are more likely to share news that aligns with their pre-existing beliefs. This "Endogenous Preference" is a strong signal for whether a piece of news is likely to be fake or real within a specific community.

### 3. The UPFD Architecture
The model uses Graph Neural Networks (GNNs) to capture two things simultaneously:
- **Structural Information (Exogenous Context):** How the news spreads from user to user in a propagation tree.
- **Content Information (Endogenous Preference):** The actual text of the news and the historical posts of the users (captured via embeddings like BERT or spaCy).

### 4. Graph Classification Task
In this project, each news story is treated as a **graph**:
- **Nodes:** The news story itself (root) and the users who shared it.
- **Edges:** "Who retweeted whom" relationships.
- **Goal:** Classify the entire graph as "Fake" or "Real".

## How to use this project
- **`config.yaml`**: The brain of the project. Toggle `use_gnn` and `use_text` to see how accuracy changes.
- **`main.py`**: The entry point to start training.
- **`src/models/`**: Explore how the GNN and Textual encoders are modularly combined.

## Further Reading
Refer to `docs/2104.12259v1.pdf` for the full technical details of the research.
