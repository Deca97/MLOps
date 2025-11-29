# Base image
FROM python:3.11-slim

WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice sorgente e CI/CD
COPY src/ ./src/
COPY tests/ ./tests/

CMD ["python", "tests/test_model.py"]
