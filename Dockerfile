# Gunakan image Python yang ringan
FROM python:3.11-slim

# Supaya Python tidak bikin file .pyc dan output log langsung keluar
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependency OS yang dibutuhkan OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set workdir di dalam container
WORKDIR /app

# Copy requirements dulu (biar layer cache efisien)
COPY requirements.txt .

# Install dependency Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode ke container
COPY . .

# Pastikan folder dataset ada
RUN mkdir -p face_dataset

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI pakai uvicorn
# GANTI `main:app` kalau nama file FastAPI kamu beda
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
