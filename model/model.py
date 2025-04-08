# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformerWithHeads(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes_task_a=3, num_classes_task_b=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.task_a_head = nn.Linear(hidden_size, num_classes_task_a)  # Sentence classification
        self.task_b_head = nn.Linear(hidden_size, num_classes_task_b)  # Another classification task

    def forward(self, input_texts, task='a'):
        encoding = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.encoder(**encoding)
        
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        
        if task == 'a':
            return self.task_a_head(pooled_output)
        elif task == 'b':
            return self.task_b_head(pooled_output)
        else:
            raise ValueError("Unknown task")

    def get_embeddings(self, input_texts):
        encoding = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.encoder(**encoding)
        return outputs.last_hidden_state[:, 0]  # CLS token
