# src/infer.py

import os, glob, re, torch
from omegaconf import OmegaConf
from torch.nn.functional import softmax
from model import get_model_and_tokenizer
from data import load_and_tokenize

def find_best_checkpoint(output_dir="output"):
    pattern = os.path.join(output_dir, "bert_epoch*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    # chọn file có epoch lớn nhất
    def epoch_num(fn):
        m = re.search(r"bert_epoch(\d+)\.pt$", fn)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=epoch_num)

def predict_texts(texts, model, tokenizer, device, max_length):
    for text in texts:
        # tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # forward
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
            label_str = "positive" if pred_id == 1 else "negative"
            print(f"Text: {text}")
            print(f" → Predicted: {label_str} (score={probs[pred_id]:.4f})\n")

if __name__ == "__main__":
    # 1. Load config
    cfg = OmegaConf.load("config/config.yaml")

    # 2. Load model & tokenizer
    ckpt = find_best_checkpoint("output")
    print(f"Loading checkpoint: {ckpt}")
    model, tokenizer = get_model_and_tokenizer(cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 3. Ví dụ để test
    examples = [
        "I absolutely loved this movie, it was fantastic!",
        "This was the worst film I've seen in years.",
        "It was okay, not great but not terrible either."
    ]

    # 4. Chạy predict và in kết quả
    predict_texts(examples, model, tokenizer, device, cfg.training.max_length)
