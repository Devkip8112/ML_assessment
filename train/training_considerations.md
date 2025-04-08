## Task 3: Training Considerations

### Scenario 1: Entire Network is Frozen
**Implications**:
- No learning occurs. Only useful if using the model strictly as a feature extractor.
- Useful for fast inference when embedding quality is known to be strong.

**When to Use**:
- Downstream tasks are very simple.
- Limited compute or few-shot learning settings.

---

### Scenario 2: Freeze Only Transformer Backbone
**Implications**:
- Task-specific heads can adapt while preserving general-purpose language features.
- Faster training, fewer parameters to update.

**When to Use**:
- Large pre-trained transformer is known to generalize well.
- Avoid catastrophic forgetting.

---

### Scenario 3: Freeze One Task Head (A or B)
**Implications**:
- Prevent interference when one task is sensitive to representation drift.
- Can stabilize a mature head while training another task.

**When to Use**:
- One task has already converged or is performing well.
- You're adding a new task without disrupting existing performance.

---

## Transfer Learning Setup

### Pre-trained Model: `bert-base-uncased`
- Widely benchmarked
- Fast and efficient for sentence-level understanding

### Freezing Strategy:
- Freeze early layers (embedding + first few transformer blocks)
- Fine-tune later layers and task heads

**Why?**
- Early layers capture syntax/universal language features
- Later layers are more task-specific
- Balanced performance + training efficiency
