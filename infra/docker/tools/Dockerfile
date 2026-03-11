FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rmse_calc_batch.py .
# Copy any shared utilities needed by rmse_calc_batch.py
COPY csv2gcp_utils.py .

ENTRYPOINT ["python", "/app/rmse_calc_batch.py"]
