import os
from typing import Dict
import csv

import fasttext
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.metrics import classification_report

from .base import Model


class FasttextClassifier(Model):
    def __init__(self,
                 config: Dict,
                 train_datapoints: pd.DataFrame = None,
                 val_datapoints: pd.DataFrame = None,
                 test_datapoints: pd.DataFrame = None):

        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.saved_model_path = os.path.join(self.base_dir, self.config["saved_model_path"])
        self.train_on = self.config["train_on"]

        if self.config["download_model"] in ["True", "true"]:
            self.model = fasttext.load_model(self.saved_model_path)
        else:
            self.model = None

        self.train_datapoints = self.process_data_for_training(train_datapoints)
        self.val_datapoints = self.process_data_for_training(val_datapoints)
        self.test_datapoints = self.process_data_for_training(test_datapoints)

    def process_data_for_training(self, df: pd.DataFrame):
        new_df = df.copy()
        new_df["processed_text"] = df[self.train_on].apply(lambda x: " ".join(simple_preprocess(x)))
        new_df["processed_label"] = df["label"].apply(lambda x: "__label__" + str(x))
        return new_df
    
    @staticmethod
    def save_data_as_txt(df: pd.DataFrame, output_file: str):
        df[["processed_label", "processed_text"]].to_csv(output_file,
                                                         index=False,
                                                         sep=" ",
                                                         header=None,
                                                         quoting=csv.QUOTE_NONE,
                                                         quotechar="",
                                                         escapechar=" ")

    @staticmethod
    def delete_file(file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    def train(self, autotune: bool = False, time_limit: int = 300):
        self.save_data_as_txt(self.train_datapoints, "data/fasttext_train.txt")
        if autotune:
            self.save_data_as_txt(self.val_datapoints, "data/fasttext_val.txt")

        if autotune:
            self.model = fasttext.train_supervised("data/fasttext_train.txt",
                                                   autotuneValidationFile='data/fasttext_val.txt',
                                                   autotuneDuration=time_limit)
        else:
            self.model = fasttext.train_supervised("data/fasttext_train.txt")

        self.delete_file("data/fasttext_train.txt")

    def save_model(self, saving_path: str = None):
        if saving_path is None:
            print(self.saved_model_path)
            self.model.save_model(self.saved_model_path)
        else:
            self.model.save_model(saving_path)

    def predict(self, test_data: pd.DataFrame):
        if "processed_text" in test_data.columns:
            if self.model is not None:
                texts = list(test_data["processed_text"])
                predictions = self.model.predict(texts)[0]
                return [int(pred[0].split("__")[2]) for pred in predictions]
            else:
                raise ValueError("You have to load or train a model")
        else:
            raise KeyError("processed_text key is missing")

    def get_metrics(self, datapoints: pd.DataFrame):
        preds = self.predict(datapoints)
        rep = classification_report(list(datapoints["label"]), preds, output_dict=True)
        rep["recall"] = rep['1']['recall']
        rep["precision"] = rep['1']['precision']
        rep["f1-score"] = rep['1']['f1-score']
        rep["support"] = [rep['0']['support'], rep['1']['support']]
        for key in ["1", "0", "macro avg", "weighted avg"]:
            rep.pop(key)
        return rep

    def compute_metrics(self, save: bool = False):
        train_metrics = self.get_metrics(self.train_datapoints)
        val_metrics = self.get_metrics(self.val_datapoints)
        test_metrics = self.get_metrics(self.test_datapoints)

        all_metrics = {"train_metrics": train_metrics,
                       "validation_metrics": val_metrics,
                       "test_metrics": test_metrics}

        # if save:
        #     with open("data/result_metrics/")

        return all_metrics
