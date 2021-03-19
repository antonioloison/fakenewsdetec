import torch
import os

from typing import Dict
from typing import List

from transformers import BertTokenizer

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict, inputs: List, labels: List):

        self.base_dir = os.path.dirname((os.path.abspath(os.path.join(__file__, os.pardir))))
        self.raw_model_path = os.path.join(self.base_dir, config["raw_model_path"])
        
        if config["download_model"] == "True":
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_type'], do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.raw_model_path, "tokenizer/"))

        self.encodings = self.tokenizer(inputs, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)