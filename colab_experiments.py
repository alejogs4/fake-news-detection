import itertools
import pandas as pd
import time
import torch
import random
import numpy as np
import os

# Ensure this script is run in an environment where main.py can be imported
from main import run_experiment

# Colab-specific imports for Google Sheets
try:
    from google.colab import auth
    from google.auth import default
    import gspread
    COLAB_ENV = True
except ImportError:
    print("Not running in Google Colab, or gspread/auth not installed. Sheet logging will be skipped/simulated.")
    COLAB_ENV = False

def setup_google_sheet(sheet_name="UPFD_Experiments_HGFND"):
    if not COLAB_ENV:
        return None
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    
    try:
        sh = gc.open(sheet_name)
        worksheet = sh.sheet1
    except gspread.exceptions.SpreadsheetNotFound:
        sh = gc.create(sheet_name)
        worksheet = sh.sheet1
        # Set up headers
        headers = ["Seed", "Dataset", "Feature", "Architecture Type", "GNN Type", "Use GNN", "Use Text", "Use HGFND (Hypergraph)", "Epochs", "Accuracy"]
        worksheet.append_row(headers)
    
    return worksheet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    worksheet = setup_google_sheet("Fake_News_Detection_HGFND_Results")
    results_file = "fake_news_detection_results.csv"
    
    # Init CSV local file
    if not os.path.exists(results_file):
        df_init = pd.DataFrame(columns=[
            "Seed", "Dataset", "Feature", "Architecture Type", "GNN Type", 
            "Use GNN", "Use Text", "Use HGFND", "Epochs", "Accuracy"
        ])
        df_init.to_csv(results_file, index=False)
        print(f"Created local results file {results_file} for logging.")
    else:
        print(f"Appending to existing local results file {results_file}.")
        
    seeds = [42, 2026]
    datasets = ["gossipcop"]
    features = ["bert"] # Can add "spacy"
    
    # 10 rigorous experiments ONLY with GNN, Text-based models, and HGFND (hiperaristas)
    # Each configuration: (architecture_name, gnn_type, use_gnn, use_text, use_hgfnd)
    experiments_matrix = [
        # 1. Text-Only baseline
        ("Text-Only", "SAGE", False, True, False),
        
        # 2-4. GNN-Only baselines (propagation trees only)
        ("GNN-Only (GCN)", "GCN", True, False, False),
        ("GNN-Only (SAGE)", "SAGE", True, False, False),
        ("GNN-Only (GAT)", "GAT", True, False, False),
        
        # 5-7. GNN + Text baselines (Early Fusion / Concatenation)
        ("GNN+Text (GCN)", "GCN", True, True, False),
        ("GNN+Text (SAGE)", "SAGE", True, True, False),
        ("GNN+Text (GAT)", "GAT", True, True, False),
        
        # 8-10. HGFND (Hypergraph Neural Network)
        ("HGFND (GCN)", "GCN", True, True, True),
        ("HGFND (SAGE)", "SAGE", True, True, True),
        ("HGFND (GAT)", "GAT", True, True, True)
    ]

    for seed in seeds:
        for dataset in datasets:
            for feature in features:
                for arch_name, gnn_type, use_gnn, use_text, use_hgfnd in experiments_matrix:
                    set_seed(seed)
                    
                    config = {
                        "data": {
                            "dataset": dataset,
                            "feature": feature,
                            "batch_size": 128,
                            "data_dir": "dataset"
                        },
                        "model": {
                            "gnn_type": gnn_type,
                            "hidden_channels": 128,
                            "use_gnn": use_gnn,
                            "use_text": use_text,
                            "use_sentiment": False,
                            "use_cmcg": False,
                            # HGFND parameters
                            "use_hgfnd": use_hgfnd,
                            "hgfnd_layers": 2,
                            "entity_clusters": 50,
                            "time_decimals": 2
                        },
                        "training": {
                            "lr": 0.001,
                            "weight_decay": 0.01,
                            "epochs": 30, # Optimized epoch length for comparative evaluation
                            "device": "auto"
                        }
                    }
                    
                    print(f"\n--- Running Experiment: {arch_name} ---")
                    print(f"Seed: {seed}, Dataset: {dataset}, Feature: {feature}")
                    
                    try:
                        acc = run_experiment(config)
                        acc_val = float(acc)
                        print(f"Achieved Accuracy: {acc_val:.4f}")
                        
                        # Log to CSV
                        row_df = pd.DataFrame([{
                            "Seed": seed, "Dataset": dataset, "Feature": feature, 
                            "Architecture Type": arch_name, "GNN Type": gnn_type, 
                            "Use GNN": use_gnn, "Use Text": use_text, "Use HGFND": use_hgfnd, 
                            "Epochs": config['training']['epochs'], "Accuracy": acc_val
                        }])
                        row_df.to_csv(results_file, mode='a', header=False, index=False)
                        
                        # Log to Google Sheets
                        if worksheet is not None:
                            row = [
                                seed, dataset, feature, arch_name, gnn_type,
                                use_gnn, use_text, use_hgfnd, 
                                config['training']['epochs'], acc_val
                            ]
                            worksheet.append_row(row)
                            time.sleep(1)
                            
                    except Exception as e:
                        print(f"Experiment failed: {e}")
                        row_df = pd.DataFrame([{
                            "Seed": seed, "Dataset": dataset, "Feature": feature, 
                            "Architecture Type": arch_name, "GNN Type": gnn_type, 
                            "Use GNN": use_gnn, "Use Text": use_text, "Use HGFND": use_hgfnd, 
                            "Epochs": config['training']['epochs'], "Accuracy": f"ERROR: {str(e)}"
                        }])
                        row_df.to_csv(results_file, mode='a', header=False, index=False)
                        
                        if worksheet is not None:
                            row = [
                                seed, dataset, feature, arch_name, gnn_type,
                                use_gnn, use_text, use_hgfnd, 
                                config['training']['epochs'], f"ERROR: {str(e)}"
                            ]
                            worksheet.append_row(row)
                            time.sleep(1)

if __name__ == "__main__":
    main()
