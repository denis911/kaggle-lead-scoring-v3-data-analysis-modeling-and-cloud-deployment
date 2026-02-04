FROM python:3.9-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . .

# Expose port (8001 changed in predict.py, but Docker usually exposes 8000 or 8080)
# We can map 8000:8001
EXPOSE 8001

# Run the application
CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8001"]
