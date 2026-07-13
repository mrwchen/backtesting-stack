FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY backtest_models ./backtest_models

CMD ["python", "-m", "src.runner"]
