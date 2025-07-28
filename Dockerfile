# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        poppler-utils \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK tokenizer data
RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng

# Copy the rest of the code
COPY . .

# Set environment variables for input and output directories
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Change working directory to the source code folder
WORKDIR /app/src

# Run the main script
CMD ["python", "model.py"]