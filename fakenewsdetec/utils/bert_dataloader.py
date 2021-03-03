import torch

from typing import Dict
from typing import List

from transformers import BertTokenizer



class BertDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict, inputs: List, labels: List):
        self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_type'], do_lower_case=True)
        self.encodings = self.tokenizer(inputs, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)