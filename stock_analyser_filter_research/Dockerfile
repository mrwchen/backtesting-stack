FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stock_analyser_filter_research ./stock_analyser_filter_research

CMD ["python", "-m", "stock_analyser_filter_research"]

