# src/model.py
from transformers import BertForSequenceClassification, BertTokenizerFast

def get_model_and_tokenizer(cfg):
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model.pretrained)
    model = BertForSequenceClassification.from_pretrained(
        cfg.model.pretrained,
        num_labels=cfg.model.num_labels
    )
    return model, tokenizer
