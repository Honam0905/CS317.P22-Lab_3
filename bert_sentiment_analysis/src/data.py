# src/data.py
from datasets import load_dataset

def load_and_tokenize(tokenizer, cfg):
    # 1) Load the IMDB splits
    raw = load_dataset(cfg.dataset.name)      # gives raw["train"] & raw["test"]

    # 2) Split off a validation set from the train split
    split = raw["train"].train_test_split(
        test_size=cfg.dataset.val_split_size, seed=42
    )
    train_ds = split["train"]
    val_ds   = split["test"]
    test_ds  = raw["test"]

    # 3) Define batch‐tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.training.max_length,
        )

    # 4) Map the tokenizer over each split
    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds   = val_ds.map(preprocess_function, batched=True)
    test_ds  = test_ds.map(preprocess_function, batched=True)

    # 5) Rename and set PyTorch format
    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")
    test_ds  = test_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format( type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds, test_ds
