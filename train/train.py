# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model.model import SentenceTransformerWithHeads

# Task A: Sentence classification
data_a = ["I love pizza", "Cats are amazing"]
labels_a = torch.tensor([0, 1])  # Example labels for classification task

# Task B: Named Entity Recognition (token classification)
data_b = ["John lives in New York", "Apple is a company"]
labels_b = [torch.tensor([1, 0, 0, 2, 3]), torch.tensor([4, 0, 0])]  # Token labels (mock NER classes)

# Model setup
model = SentenceTransformerWithHeads()
loss_fn_a = nn.CrossEntropyLoss()
loss_fn_b = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

model.train()

# Forward pass for task A
outputs_a = model(data_a, task='a')
loss_a = loss_fn_a(outputs_a, labels_a)

# Forward pass for task B
encoding_b = model.tokenizer(data_b, padding=True, truncation=True, return_tensors="pt")
outputs_b = model.encoder(**encoding_b).last_hidden_state  # [batch, seq, hidden]
logits_b = model.task_b_head(outputs_b)  # [batch, seq, classes]

from torch.nn.utils.rnn import pad_sequence
labels_b = [torch.tensor([1, 0, 0, 2, 3]), torch.tensor([4, 0, 0])]
labels_b_padded = pad_sequence(labels_b, batch_first=True, padding_value=-100)  # [batch, seq]

# Ensure shapes match before computing loss
logits_b = logits_b[:, :labels_b_padded.size(1), :]  # truncate if needed
loss_b = loss_fn_b(logits_b.reshape(-1, logits_b.size(-1)), labels_b_padded.view(-1))

# Total loss and backward
loss = loss_a + loss_b
loss.backward()
optimizer.step()

print("Loss A:", loss_a.item())
print("Loss B:", loss_b.item())
print("Total Loss:", loss.item())

