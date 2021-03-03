
from typing import Dict
from typing import List

import torch
import pandas as pd

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from utils.bert_dataloader import BertDataset

class BertModel():
    def __init__(self, config: Dict):
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(config['model_type'], num_labels = 2)
        self.training_args = TrainingArguments(**config["training_args"])
        
    
    def train(self,
              train_datapoints: pd.DataFrame,
              val_datapoints: pd.DataFrame,
    ):
        inputs_train = train_datapoints[self.config['train_on']].values.tolist()
        labels_train = train_datapoints['label'].values.tolist()

        inputs_eval = val_datapoints[self.config['train_on']].values.tolist()
        labels_eval = val_datapoints['label'].values.tolist()

        train_data = BertDataset(self.config, inputs_train, labels_train)
        eval_data = BertDataset(self.config, inputs_eval, labels_eval)
       
        self.trainer = Trainer(model=self.model, args=self.training_args, train_dataset=train_data, eval_dataset=eval_data)
        self.trainer.train()
""" 
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        expected_labels = [datapoint.label for datapoint in eval_datapoints]
        predicted_proba = self.predict(eval_datapoints)
        predicted_labels = np.argmax(predicted_proba, axis=1)
       
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        print(f"Accuracy: {accuracy}, F1: {f1}, AUC: {auc}")
        split_prefix = "" if split is None else split
        return {
            f"{split_prefix} f1": f1,
            f"{split_prefix} accuracy": accuracy,
            f"{split_prefix} auc": auc,
            f"{split_prefix} true negative": tn,
            f"{split_prefix} false negative": fn,
            f"{split_prefix} false positive": fp,
            f"{split_prefix} true positive": tp,
        }
    
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        data = FakeNewsTorchDataset(self.config, datapoints)
        dataloader = DataLoader(data,
                                batch_size=self.config["batch_size"],
                                pin_memory=True)
        self.model.eval()
        predicted = []
        self.model.cuda()
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                output = self.model(input_ids=batch["ids"].cuda(),
                                    attention_mask=batch["attention_mask"].cuda(),
                                    token_type_ids=batch["type_ids"].cuda(),
                                    labels=batch["label"].cuda())
                predicted.append(output[1])
        return torch.cat(predicted, axis=0).cpu().detach().numpy()
    
    def get_params(self) -> Dict:
        return {}
"""