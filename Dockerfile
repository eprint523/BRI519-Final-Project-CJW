# Mouse LFP Analysis - Docker Configuration
# Author: Choi Joung Woo
# Course: BRI519 (Fall 2025)

# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY main.py .

# Create output directory
RUN mkdir -p /app/output

# Set matplotlib backend to Agg (non-interactive)
ENV MPLBACKEND=Agg

# Run the analysis
CMD ["python", "main.py"]
