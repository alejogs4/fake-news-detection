import itertools
import pandas as pd
import time
import torch
import random
import numpy as np

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

def setup_google_sheet(sheet_name="UPFD_Experiments"):
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
        headers = ["Seed", "Dataset", "Feature", "Use GNN", "GNN Type", "Use Text", "Use CMCG (Co-Attention)", "Use Sentiment", "Epochs", "Accuracy"]
        worksheet.append_row(headers)
    
    return worksheet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    worksheet = setup_google_sheet("Fake_News_Detection_Results")
    
    # Define the test matrix
    seeds = [42, 2026]
    datasets = ["gossipcop"] # Can add "politifact"
    features = ["bert"] # or "spacy"
    
    # Combinations of model architectures
    # Each tuple: (use_gnn, gnn_type, use_text, use_cmcg, use_sentiment)
    model_configs = [
        # Baseline Text only
        (False, "GCN", True, False, False),
        
        # Baseline GNN only (GCN & SAGE)
        (True, "GCN", False, False, False),
        (True, "SAGE", False, False, False),
        
        # GNN + Text (Early fusion / Concatenation)
        (True, "GCN", True, False, False),
        (True, "SAGE", True, False, False),
        
        # CMCG (Co-Attention between GNN and Text)
        (True, "SAGE", True, True, False),
        
        # Text + Sentiment
        (False, "GCN", True, False, True),
        
        # Full Model: GNN + Text + CMCG + Sentiment
        (True, "SAGE", True, True, True),
    ]

    for seed in seeds:
        for dataset in datasets:
            for feature in features:
                for use_gnn, gnn_type, use_text, use_cmcg, use_sentiment in model_configs:
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
                            "use_cmcg": use_cmcg,
                            "use_sentiment": use_sentiment,
                            "sentiment_dim": 1 # Example dummy sentiment dimension
                        },
                        "training": {
                            "lr": 0.001,
                            "weight_decay": 0.01,
                            "epochs": 30, # Reduced for testing, adjust as needed
                            "device": "auto"
                        }
                    }
                    
                    print(f"\n--- Running Experiment ---")
                    print(f"Seed: {seed}, Dataset: {dataset}, Feature: {feature}")
                    print(f"GNN: {use_gnn} ({gnn_type}), Text: {use_text}, CMCG: {use_cmcg}, Sentiment: {use_sentiment}")
                    
                    try:
                        acc = run_experiment(config)
                        print(f"Achieved Accuracy: {acc:.4f}")
                        
                        # Log to Google Sheets
                        if worksheet is not None:
                            row = [
                                seed, dataset, feature, 
                                use_gnn, gnn_type, use_text, use_cmcg, use_sentiment, 
                                config['training']['epochs'], float(acc)
                            ]
                            worksheet.append_row(row)
                            # Sleep briefly to avoid Google Sheets API rate limits
                            time.sleep(1)
                    except Exception as e:
                        print(f"Experiment failed: {e}")
                        if worksheet is not None:
                            row = [
                                seed, dataset, feature, 
                                use_gnn, gnn_type, use_text, use_cmcg, use_sentiment, 
                                config['training']['epochs'], f"ERROR: {str(e)}"
                            ]
                            worksheet.append_row(row)

if __name__ == "__main__":
    main()
