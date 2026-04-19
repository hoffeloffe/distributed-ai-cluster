FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash aiuser
USER aiuser
WORKDIR /app
COPY --from=builder /root/.local /home/aiuser/.local
ENV PATH=/home/aiuser/.local/bin:$PATH
COPY --chown=aiuser:aiuser . .
EXPOSE 8080 8888
CMD ["python", "-m", "src.entrypoint"]
