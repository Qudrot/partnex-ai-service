# Use the lightweight Python 3.11 image to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by XGBoost and machine learning libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker's layer caching
# (This prevents reinstalling packages every time you change your code)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your AI backend code into the container
COPY . .

# Expose the port your API will run on
EXPOSE 10000

# Start the server using your exact Gunicorn configuration
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 1 --timeout 120
