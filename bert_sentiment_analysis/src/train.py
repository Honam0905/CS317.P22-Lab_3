# src/train.py
import os
import mlflow
import torch
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from model import get_model_and_tokenizer
from data import load_and_tokenize

def setup_mlflow(cfg):
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(cfg.logging.mlflow_experiment)

def train():
    cfg = OmegaConf.load("config/config.yaml")
    setup_mlflow(cfg)
    with mlflow.start_run():
        # 1. Log params
        mlflow.log_params({
            "epochs":       cfg.training.epochs,
            "batch_size":   cfg.training.batch_size,
            "lr":           cfg.training.lr,
            "max_length":   cfg.training.max_length,
            "dataset":      cfg.dataset.name,
            "model":        cfg.model.pretrained
        })

        # 2. Prepare data & model
        model, tokenizer = get_model_and_tokenizer(cfg)
        train_ds, val_ds, test_ds = load_and_tokenize(tokenizer, cfg)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg.training.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 3. Optimizer + scheduler
        optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        best_val_acc = 0.0
        # 4. Training loop
        for epoch in range(1, cfg.training.epochs+1):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {k:v.to(device) for k,v in batch.items() if k!="labels"}
                labels = batch["labels"].to(device)
                loss = model(**inputs, labels=labels).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # 5. Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k:v.to(device) for k,v in batch.items() if k!="labels"}
                    labels = batch["labels"].to(device)
                    logits = model(**inputs).logits
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds==labels).sum().item()
                    total   += labels.size(0)
            val_acc = correct/total

            # 6. Log metrics
            mlflow.log_metrics({
                f"train_loss_epoch_{epoch}": avg_train_loss,
                f"val_acc_epoch_{epoch}":    val_acc
            }, step=epoch)

            # 7. Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = f"output/bert_epoch{epoch}.pt"
                os.makedirs("output", exist_ok=True)
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(ckpt, artifact_path="checkpoints")

        # 8. Return test dataset for later evaluation
        return cfg, model, tokenizer, test_ds

if __name__ == "__main__":
    train()
