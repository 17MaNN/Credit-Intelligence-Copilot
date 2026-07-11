FROM python:3.11-slim AS base
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY lib /app/lib
COPY requirements.base.txt .
RUN pip install --no-cache-dir -r requirements.base.txt

FROM base
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY services/doc-cv/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/doc-cv/ .
RUN python train.py

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]