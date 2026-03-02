# ---- Base Image ----
FROM python:3.11-slim
 
# ---- Environment Settings ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
 
# ---- Set Work Directory ----
WORKDIR /app
 
# ---- Install System Dependencies (needed for numpy/sklearn) ----
RUN apt-get update && apt-get install -y \
    build-essential \
&& rm -rf /var/lib/apt/lists/*
 
# ---- Copy Requirements First (for Docker layer caching) ----
COPY requirements.txt .
 
# ---- Install Python Dependencies ----
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# ---- Copy Application Code ----
COPY . .
 
# ---- Expose Port ----
EXPOSE 8000
 
# ---- Start Application ----
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8000"]
