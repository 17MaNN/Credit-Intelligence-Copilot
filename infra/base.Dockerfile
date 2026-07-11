FROM python:3.11-slim AS base
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY lib /app/lib
COPY requirements.base.txt .
RUN pip install --no-cache-dir -r requirements.base.txt
USER appuser
