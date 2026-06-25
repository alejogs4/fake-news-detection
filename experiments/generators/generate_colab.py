import json
import os

def create_writefile_cell(filepath, content):
    # Ensure directory exists in Colab
    dir_name = os.path.dirname(filepath)
    mkdir_code = f"!mkdir -p {dir_name}\n" if dir_name else ""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": mkdir_code + f"%%writefile {filepath}\n" + content
    }

def create_code_cell(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content
    }

def create_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content
    }

files_to_write = [
    "src/utils.py",
    "src/data.py",
    "src/trainer.py",
    "src/models/cmcg.py",
    "src/models/classifier.py",
    "src/models/text_encoders.py",
    "src/models/gnn_encoders.py",
    "src/models/hgfnd.py",
    "src/models/upfd_model.py",
    "main.py",
    "config.yaml",
    "colab_experiments.py"
]

cells = []

cells.append(create_markdown_cell(["# Fake News Detection - Standalone Environment\n", "Run the cells below sequentially. The first few cells will automatically generate the project structure inside your Colab environment."]))

for file in files_to_write:
    with open(file, 'r') as f:
        content = f.read()
    cells.append(create_writefile_cell(file, content))

cells.append(create_markdown_cell(["# 1. Install Dependencies"]))
cells.append(create_code_cell(["!pip install torch_geometric PyYAML gspread"]))

cells.append(create_markdown_cell(["# 2. Download and Extract Dataset\n", "Fixes the 404 error from PyTorch Geometric."]))
cells.append(create_code_cell([
    "!mkdir -p dataset/gossipcop/raw\n",
    "!curl -L -o dataset/gossipcop/raw/data.zip \"https://data.pyg.org/datasets/upfd_gossipcop.zip\"\n",
    "!cd dataset/gossipcop/raw && unzip -o data.zip && rm data.zip"
]))

cells.append(create_markdown_cell(["# 3. Run Experiments\n", "This will authenticate with Google Sheets and start the training loops."]))
cells.append(create_code_cell(["!python colab_experiments.py"]))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("fake_news_colab_standalone.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Notebook generated successfully.")
