FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends bash git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    openhands-sdk==1.49.6 openhands-tools==1.49.6 \
    pandas==2.3.3 openpyxl==3.1.5 xlrd==2.0.2 \
    duckdb==1.5.4 Pillow==12.3.0 \
    pydicom==3.0.1 pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.3.0 \
    pylibjpeg-openjpeg==2.5.0 nibabel==5.3.2 pynrrd==1.1.3 SimpleITK==2.5.3
COPY src/medbenchagent/runtime/openhands_worker.py /app/openhands_worker.py
WORKDIR /workspace
