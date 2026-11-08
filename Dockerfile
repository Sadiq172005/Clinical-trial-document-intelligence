FROM python:3.11-slim

# tesseract-ocr + poppler-utils needed for scanned-page OCR / pdf2image fallback in pipeline.py
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py app.py ./
COPY data/trial_docs ./data/trial_docs

ENV TRIAL_DOCS_DIR=/app/data/trial_docs
ENV INDEX_DIR=/app/data/faiss_index
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
