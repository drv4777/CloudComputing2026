FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pandas matplotlib seaborn

CMD ["python", "data_analysis.py"]
