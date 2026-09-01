# docker/Dockerfile
# Production Docker image for the Kubernetes Scheduler Simulator (Python port)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional, for potential future needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for configs, workloads, and results if they don't exist
RUN mkdir -p configs workloads results

# Set the entrypoint to the CLI
ENTRYPOINT ["python", "main.py"]

# Default command (can be overridden)
CMD ["run", "--help"]