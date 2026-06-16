FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY src/ ./src/
COPY README.md LICENSE ./

# Install dependencies (no dev for production)
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction

# Create non-root user
RUN useradd -m -u 1000 distill && chown -R distill:distill /app
USER distill

# Set entry point
ENTRYPOINT ["distill-align"]
CMD ["--help"]
