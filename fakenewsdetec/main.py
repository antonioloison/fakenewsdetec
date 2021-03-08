import os
import argparse
import json
import logging
import pandas as pd

import numpy as np
import torch

from model.bert import BertModel

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   
    train_data_path = os.path.join(base_dir, config["train_data_path"])
    val_data_path = os.path.join(base_dir, config["val_data_path"])
    test_data_path = os.path.join(base_dir, config["test_data_path"])
    
    # Read data
    train_datapoints = pd.read_csv(train_data_path)
    val_datapoints = pd.read_csv(val_data_path)
    test_datapoints = pd.read_csv(test_data_path)
    
    if config["model"] == "bert":
        print("Loading Model...")
        model = BertModel(config, train_datapoints, val_datapoints, test_datapoints)
        if config["train"] == "False":
            print(" Model is loaded from an already-trained previous backup")
            print("Predictions result: ")
            model.compute_metrics()       
        else:
            print(" Training Model...")
            model.train()
            print(" Evaluating Model...")
            mode.evaluate()

    