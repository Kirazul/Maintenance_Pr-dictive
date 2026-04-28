FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/api/requirements.txt

COPY . /app

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
