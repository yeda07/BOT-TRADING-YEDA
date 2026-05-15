FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python -m app.deployment.docker_healthcheck

CMD ["python", "-m", "app.main", "healthcheck"]
