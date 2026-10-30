FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EXPRESSION_LAB_HOME=/data
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --require-hashes -r requirements-api.txt && pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 expression && mkdir -p /data/models && chown -R expression:expression /data
USER expression
VOLUME ["/data"]
ENTRYPOINT ["expression-lab"]
CMD ["--check"]
