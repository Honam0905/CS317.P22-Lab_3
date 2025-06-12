FROM python:3.9-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      git-lfs \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serve/requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY output/ /app/output/
COPY serve/app.py /app/app.py

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]