# src/evaluate.py

import os
import glob
import re
import torch
import mlflow
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from model import get_model_and_tokenizer
from data import load_and_tokenize

def find_best_checkpoint(output_dir="output"):
    # find all files matching bert_epoch{N}.pt
    pattern = os.path.join(output_dir, "bert_epoch*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    # extract epoch number and pick the largest
    def epoch_num(fn):
        m = re.search(r"bert_epoch(\d+)\.pt$", fn)
        return int(m.group(1)) if m else -1
    best_ckpt = max(files, key=epoch_num)
    return best_ckpt

def evaluate():
    # 1. Load config
    cfg = OmegaConf.load("config/config.yaml")

    # 2. Locate best checkpoint
    ckpt_path = find_best_checkpoint("output")
    print(f"Loading checkpoint: {ckpt_path}")

    # 3. Load model & tokenizer
    model, tokenizer = get_model_and_tokenizer(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 4. Prepare test data
    _, _, test_ds = load_and_tokenize(tokenizer, cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size)

    # 5. Run evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f}")

    # 6. (Optional) Log to MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(cfg.logging.mlflow_experiment + "_eval")
    with mlflow.start_run():
        mlflow.log_param("loaded_checkpoint", os.path.basename(ckpt_path))
        mlflow.log_metric("test_accuracy", test_acc)

    return test_acc

if __name__ == "__main__":
    evaluate()
