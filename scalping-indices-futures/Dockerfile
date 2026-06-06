FROM python:3.13-slim
# Single-threaded BLAS: each layer (HMM / GARCH / state-space) is fit per-bar in a
# walk-forward loop, so we avoid oversubscribing cores with internal threads.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scalp_core/ ./scalp_core/
COPY backtest_runner.py .
CMD ["python", "backtest_runner.py"]
