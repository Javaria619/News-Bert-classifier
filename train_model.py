from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load dataset
dataset = load_dataset("ag_news")

# Reduce dataset size (FAST training)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(5000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

# Tokenization
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# Rename label column
dataset = dataset.rename_column("label", "labels")

# Torch format
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

# Metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Training arguments (CPU friendly)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=100,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

print("✅ DistilBERT training completed FAST!")
