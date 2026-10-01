FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY production_pipeline.py week3_day2_api.py week3_day3_dashboard.py ./
COPY final_production_candidate.joblib final_modeling_dataset.csv ./

EXPOSE 8000 8501

CMD ["uvicorn", "week3_day2_api:app", "--host", "0.0.0.0", "--port", "8000"]
