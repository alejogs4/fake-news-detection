# User-Preference-aware Fake News Detection (UPFD)

This project is a modular, research-oriented implementation of the UPFD framework. It explores the intersection of social psychology and deep learning to detect fake news by focusing on **who** spreads the news, not just **what** the news says.

---

## 🧠 The Theory: Why this works

Traditional fake news detection focuses on the content of the news (Exogenous signals). However, this project implements the research from **"User-Preference-aware Fake News Detection" (SIGIR 2021)**, which leverages:

### 1. Confirmation Bias
Users tend to believe and share information that aligns with their pre-existing beliefs, regardless of truth. This is known as **Endogenous Preference**.

### 2. Endogenous vs. Exogenous Signals
- **Exogenous Signals (News Content):** The raw text of the article.
- **Endogenous Signals (User Preference):** The historical behavior and preferences of the users sharing the news.
- **Propagation Context:** How the news spreads through a social network (the tree structure).

### 3. How the Model Combines Them
- **Graph Neural Networks (GNN):** Capture the "Exogenous Context" by analyzing the propagation tree. It learns features from how the information flows between users.
- **Textual Encoders:** Capture the "Endogenous Preference" by analyzing the content of the news and user features.
- **The Result:** By concatenating these two views, the model identifies "fake" news not just by spotting lies, but by identifying the specific "echo chambers" and biased sharing patterns typical of misinformation.

---

## 🚀 Getting Started

### 1. Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
- Python 3.10+.

### 2. Installation
Clone the repository and set up the environment:
```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate fake-news-detection

# Install the project in editable mode
pip install -e .
```

### 3. Running the Project
The project is configuration-driven. You don't need to change code to run different experiments.

**Run with default settings:**
```bash
python main.py
```

**Experimenting:**
Open `config.yaml` to:
- Switch datasets: `politifact` or `gossipcop`.
- Change GNN backbones: `GCN`, `GAT`, or `SAGE`.
- Toggle features: Set `use_gnn: false` or `use_text: false` to see the impact of each branch.

---

## ☁️ Cloud Deployment
This project is built to be "Cloud-Ready." For detailed instructions on deploying to **Google Cloud Platform (Vertex AI)** using Docker, please refer to:
👉 **[docs/cloud_deploy.md](docs/cloud_deploy.md)**

---

## 📚 Learning More
For a deeper dive into the architecture and the SIGIR paper, check out our internal guide:
👉 **[docs/learning_guide.md](docs/learning_guide.md)**
