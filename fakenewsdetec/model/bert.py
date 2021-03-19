import os
import torch
import pandas as pd
import numpy as np

from typing import Dict
from typing import List
from typing import Optional

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from utils.bert_dataloader import BertDataset
from .base import Model


class BertModel(Model):
    def __init__(self, 
                config: Dict, 
                train_datapoints: pd.DataFrame,
                val_datapoints: pd.DataFrame,
                test_datapoints: pd.DataFrame,
    ):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.saved_model_path = os.path.join(self.base_dir, self.config["saved_model_path"])
        self.raw_model_path = os.path.join(self.base_dir, self.config["raw_model_path"])

        if self.config["train"] == "False":           
            self.model = BertForSequenceClassification.from_pretrained(self.saved_model_path)
        
        elif self.config["download_model"] == "True":
            self.model = BertForSequenceClassification.from_pretrained(self.config['model_type'], num_labels = 2)
        
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.raw_model_path, num_labels = 2)

        self.training_args = TrainingArguments(**config["training_args"])

        self.inputs_train = train_datapoints[self.config['train_on']].values.tolist()
        self.labels_train = train_datapoints['label'].values.tolist()

        self.inputs_eval = val_datapoints[self.config['train_on']].values.tolist()
        self.labels_eval = val_datapoints['label'].values.tolist()

        self.inputs_test = test_datapoints[self.config['train_on']].values.tolist()
        self.labels_test = test_datapoints['label'].values.tolist()

        self.train_data = BertDataset(self.config, self.inputs_train, self.labels_train)
        self.eval_data = BertDataset(self.config, self.inputs_eval, self.labels_eval)
        self.test_data = BertDataset(self.config, self.inputs_test, self.labels_test)
       
        self.trainer = Trainer(model=self.model, args=self.training_args, train_dataset=self.train_data, eval_dataset=self.eval_data)
          
    def train(self):
        
        self.trainer.train()
        self.trainer.save_model(self.saved_model_path)
    
    def eval(self):
        print(self.trainer.evaluate())

    def predict(self, test_data):
        return self.trainer.predict(test_data)
        
    def compute_metrics(self):
        expected_labels = self.labels_test
        predicted_proba = self.predict(self.test_data)
        predicted_labels = np.argmax(predicted_proba[0], axis=1)
       
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_labels)
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        print(f"Accuracy: {accuracy}, F1: {f1}, AUC: {auc}")

        print( {"true negative": tn,
                "false negative": fn,
                "false positive": fp,
                "true positive": tp,
        })