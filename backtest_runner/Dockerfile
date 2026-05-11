FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backtest_models/ ./backtest_models/
COPY backtest_runner.py .
CMD ["python", "backtest_runner.py"]
