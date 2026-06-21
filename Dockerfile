FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependências de sistema mínimas (psycopg2, build de wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências primeiro para aproveitar o cache de camadas
COPY pyproject.toml ./
COPY README.md ./
# Instala o torch na variante CPU-only (a VM não tem GPU): evita baixar o
# stack CUDA da NVIDIA (~vários GB), reduzindo o tamanho da imagem.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

COPY . .

EXPOSE 8000 8501
