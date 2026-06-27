FROM python:3.13-slim

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hfmed_range_analysis/ ./hfmed_range_analysis/
COPY range_analysis_runner.py .
CMD ["python", "range_analysis_runner.py"]

