import torch

from typing import Dict
from typing import List

from transformers import BertTokenizer

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict, inputs: List, labels: List):
        
        if config["download_model"] == "True" :
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_type'], do_lower_case=True)
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.raw_model_path = os.path.join(base_dir, config["raw_model_path"])
            self.tokenizer = BertTokenizer.from_pretrained(config["raw_model_path"], do_lower_case=True)
       
        self.encodings = self.tokenizer(inputs, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)