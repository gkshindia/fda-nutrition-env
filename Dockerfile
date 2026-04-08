FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy lock file and manifest first for layer caching
COPY pyproject.toml uv.lock ./

# Copy all application code
COPY data/ data/
COPY core/ core/
COPY env/ env/
COPY baseline.py .
COPY openenv.yaml .

# Install dependencies (locked)
RUN uv sync --no-dev --frozen

ENV PATH="/app/.venv/bin:$PATH"

# HF Spaces requires non-root uid 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
