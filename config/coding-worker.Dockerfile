FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends bash git \
    && rm -rf /var/lib/apt/lists/*
COPY config/coding-worker.requirements.txt /tmp/coding-worker.requirements.txt
RUN pip install --no-cache-dir -r /tmp/coding-worker.requirements.txt
COPY src/medbenchagent/runtime/openhands_worker.py /app/openhands_worker.py
COPY src/medbenchagent /app/medbenchagent
WORKDIR /workspace
