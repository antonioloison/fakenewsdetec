import os
import argparse
import json
import pandas as pd

from fakenewsdetec.model.bert import BertModel
from model.fasttext_classifier import FasttextClassifier
from utils.dataset_loader import Dataset


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   
    dataset = Dataset('../data')
    # Read data
    train_datapoints, val_datapoints, test_datapoints = dataset.load_dataset(['berkeley'])

    
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
            model.eval()

    elif config["model"] == "fasttext":
        print("Loading Model...")
        model = FasttextClassifier(config, train_datapoints, val_datapoints, test_datapoints)
        if bool(config["train"]):
            print("Training Model...")
            model.train()
        print("Model is loaded from an already-trained previous backup")
        print("Predictions result: ")
        print(model.compute_metrics())
