# src/tune.py

import os
import mlflow
import torch
import optuna
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from data import load_and_tokenize

def setup_mlflow(experiment_name: str):
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(experiment_name)

def objective(trial):
    # 1. Load base config
    cfg = OmegaConf.load("configs/config.yaml")
    
    # 2. Sample hyperparameters
    cfg.training.lr = trial.suggest_loguniform("lr", 1e-6, 5e-5)
    cfg.training.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    cfg.training.epochs = trial.suggest_int("epochs", 2, 4)
    
    # 3. Start an MLflow run per trial
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
        mlflow.log_params({
            "lr": cfg.training.lr,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs
        })
        
        # 4. Prepare model & data
        tokenizer = BertTokenizerFast.from_pretrained(cfg.model.pretrained)
        model = BertForSequenceClassification.from_pretrained(
            cfg.model.pretrained,
            num_labels=cfg.model.num_labels
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_ds, val_ds, _ = load_and_tokenize(tokenizer, cfg)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size)

        # 5. Optimizer & scheduler
        optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # 6. Training loop
        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                loss = model(**inputs, labels=labels).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # 7. Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        mlflow.log_metric("val_accuracy", val_acc)

        return val_acc

if __name__ == "__main__":
    # 0. Khởi tạo MLflow experiment
    base_cfg = OmegaConf.load("configs/config.yaml")
    setup_mlflow(base_cfg.logging.mlflow_experiment + "_optuna")

    # 1. Tạo study và chạy
    study = optuna.create_study(direction="maximize",
                                study_name="bert_sentiment_optuna")
    study.optimize(objective, n_trials=10)

    # 2. In kết quả
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
    
    # 3. Ghi best params vào file để bạn dễ load lần sau
    with open("best_params.yaml", "w") as fp:
        OmegaConf.save(OmegaConf.create({"training": trial.params}), fp)
