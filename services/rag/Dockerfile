FROM python:3.11-slim AS base
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY lib /app/lib
COPY requirements.base.txt .
RUN pip install --no-cache-dir -r requirements.base.txt

FROM base
COPY services/rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/rag/ .
RUN python build_index.py && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]