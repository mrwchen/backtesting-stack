FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY swing_stock_backtester ./swing_stock_backtester

CMD ["python", "-m", "swing_stock_backtester.runner"]
