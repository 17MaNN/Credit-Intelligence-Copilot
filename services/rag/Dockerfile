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

ENV HF_HUB_DOWNLOAD_TIMEOUT=60
RUN (python build_index.py || (echo "build_index.py failed, retrying once..." && sleep 10 && python build_index.py)) \
    && chown -R appuser:appuser /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]