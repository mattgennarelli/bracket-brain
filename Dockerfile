FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] aiofiles

# Copy application
COPY engine.py api.py backtest.py run.py ./
COPY scripts/ scripts/
COPY web/ web/
COPY data/ data/

# Expose port
EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
