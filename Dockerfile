FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backtest_shared.py .
COPY backtest_core/ ./backtest_core/
COPY backtest_models/ ./backtest_models/
COPY backtest_model_configs/ ./backtest_model_configs/
COPY backtest_runner.py .
CMD ["python", "backtest_runner.py"]
