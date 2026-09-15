FROM python:3.11.11-slim
RUN pip install --no-cache-dir python-dotenv==1.1.1 duckdb==1.5.4 pydantic==2.13.4 pandas==2.3.3 openpyxl==3.1.5 xlrd==2.0.2 Pillow==11.3.0 pydicom==3.0.1 numpy==2.2.6 nibabel==5.3.2 pynrrd==1.1.3 SimpleITK==2.5.3 pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.3.0 pylibjpeg-openjpeg==2.5.0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /work
