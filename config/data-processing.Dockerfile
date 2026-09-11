FROM python:3.11-slim
RUN pip install --no-cache-dir pandas openpyxl Pillow pydicom numpy nibabel pynrrd SimpleITK
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /work
