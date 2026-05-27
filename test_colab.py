import sys
import copy
from colab_experiments import main, experiments_matrix

# override experiments matrix to only run HGFND (GCN) for politifact 1 seed
import colab_experiments
colab_experiments.experiments_matrix = [("HGFND (GCN)", "GCN", True, True, True)]
colab_experiments.seeds = [42]
colab_experiments.datasets = ["politifact"]

colab_experiments.main()
