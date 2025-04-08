# README.md

# ML Apprentice Take-Home Assessment

This repository contains a solution to a take-home machine learning assessment focusing on Sentence Transformers and Multi-Task Learning.

## Project Structure
```
.
├── model/               # Sentence Transformer model with multi-task heads
├── train/               # Mock training logic with sample forward passes
├── requirements.txt     # Dependencies
├── Dockerfile           # Container setup
└── README.md            # Documentation
```

## Task Overview

### Task 1: Sentence Transformer
Implemented a model that encodes sentences into fixed-length embeddings using the CLS token from a `bert-base-uncased` backbone. Used HuggingFace Transformers.

### Task 2: Multi-Task Expansion
The model supports two tasks:
- **Task A**: Sentence Classification — classifies sentences into categories (e.g., topic or intent).
- **Task B**: Named Entity Recognition (NER) — a token-level task where each token is labeled (e.g., PER, LOC, ORG).

The architecture was adapted so that:
- Task A uses the `[CLS]` token for sentence-level output.
- Task B uses the full `last_hidden_state` for token-level classification.

Mock token-level labels were padded and flattened for loss computation using `CrossEntropyLoss` with an ignore index of `-100`.


### Task 3: Training Considerations
Covered scenarios for freezing different parts of the network. Suggested a transfer learning strategy that fine-tunes later transformer layers and head layers for adaptability while preserving general language representations.

### Task 4 (Bonus): Training Loop
Implemented mock training logic:
- Forward passes for both tasks
- CrossEntropy losses per task
- Combined optimization step

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the training script from the project root
python -m train.train
```

## Docker Usage
```bash
docker build -t ml-apprentice .
docker run --rm ml-apprentice
```

## Notes
- Sample data is synthetic
- Evaluation metrics and datasets can be plugged in easily
- This serves as a proof of concept for a multi-task learning setup

---

**Author**: Devki Patel
