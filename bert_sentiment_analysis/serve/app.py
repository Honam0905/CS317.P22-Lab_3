import os
import glob
import re
from pathlib import Path
import subprocess, sys
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from torch.nn.functional import softmax
from transformers import BertTokenizerFast, BertForSequenceClassification
import logging, time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str
    score: float


def find_best_ckpt(output_dir: str):
    pattern = os.path.join(output_dir, "bert_epoch*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    def epoch_num(fn):
        m = re.search(r"bert_epoch(\d+)\.pt$", fn)
        return int(m.group(1)) if m else -1
    return max(files, key=epoch_num)


# --- determine where "output/" actually lives ---
THIS_DIR    = Path(__file__).resolve().parent        # e.g. /app/serve (locally) or /app (in Docker)
PROJECT_DIR = THIS_DIR.parent                       # one level up

# first try the project-root location
OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", str(PROJECT_DIR / "output")))

# if that path doesn't exist (e.g. inside container you only have /app/output),
# then fall back to a sibling "output/" next to this file
if not OUTPUT_DIR.exists():
    alt = THIS_DIR / "output"
    if alt.exists():
        OUTPUT_DIR = alt


CKPT = find_best_ckpt(OUTPUT_DIR)


# --- load model & tokenizer ---
PRETRAINED = "bert-base-uncased"
NUM_LABELS = 2

tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED, num_labels=NUM_LABELS
)
state = torch.load(CKPT, map_location="cpu",weights_only=False)
model.load_state_dict(state)
model.eval()


app = FastAPI(title="BERT Sentiment API")


# Định nghĩa logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s - %(message)s')
logger = logging.getLogger("sentiment_api")

# --- custom Prometheus metrics ---
INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Thời gian inference mô hình BERT (giây)"
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence_score",
    "Phân phối confidence score của mô hình"
)
REQUEST_COUNTER = Counter(
    "predict_requests_total",
    "Tổng số request tới /predict",
    ["status"]
)

# đăng ký instrumentator
instrumentator = Instrumentator().instrument(app).expose(app)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
      <head>
        <title>Sentiment Analyzer</title>
      </head>
      <body>
        <h1>Enter text for Sentiment Analysis</h1>
        <form action="/analyze" method="post">
          <textarea name="text" rows="4" cols="60"
                    placeholder="Type your text here..."></textarea><br>
          <button type="submit">Analyze</button>
        </form>
      </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(text: str = Form(...)):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        label = "positive" if idx == 1 else "negative"
        score = probs[idx].item()

    return f"""
    <html>
      <head><title>Result</title></head>
      <body>
        <h1>Sentiment Analysis Result</h1>
        <p><strong>Text:</strong> {text}</p>
        <p><strong>Sentiment:</strong> {label}</p>
        <p><strong>Confidence:</strong> {score:.4f}</p>
        <a href="/">Analyze another text</a>
      </body>
    </html>
    """


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    start = time.time()
    try:
        inputs = tokenizer(
            payload.text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
            label = "positive" if idx == 1 else "negative"
            score = float(probs[idx])

        PREDICTION_CONFIDENCE.observe(score)
        REQUEST_COUNTER.labels(status="success").inc()
        return PredictionOut(label=label, score=score)
    except Exception as e:
        logger.exception("Error during prediction")
        REQUEST_COUNTER.labels(status="error").inc()
        raise e
    finally:
        INFERENCE_TIME.observe(time.time() - start)