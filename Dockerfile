FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY config.example.yaml ./config.example.yaml
EXPOSE 8000

# The generation/embedding servers run outside this image, so a Mac can use
# Metal GPU acceleration without GPU pass-through from Docker Desktop.
CMD ["uvicorn", "localrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
