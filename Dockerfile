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

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).from
# lets us import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

#  Expose FastAPI port
EXPOSE 7860

# Run the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860"]