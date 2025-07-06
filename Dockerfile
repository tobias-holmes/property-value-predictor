# Use official Python image as base
FROM python:3.11-slim

# Set working dir within the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements-docker.txt /app
RUN pip install -r requirements-docker.txt

# Copy relevant files into the container
COPY src/ /app/src/
COPY models/ /app/models/

# Expose port 8080
EXPOSE 80

# Set PYTHONPATH
ENV PYTHONPATH=src

# Run prediction service
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]