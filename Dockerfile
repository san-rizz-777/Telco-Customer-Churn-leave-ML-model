# python docker image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy only dependencies file first for Docker caching.
COPY requirements.txt .

# Install the python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the image
COPY . .

# Explicitly copy model (in case .dockerignore excluded mlruns)
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets us import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

#  Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]