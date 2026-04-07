FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app
COPY pyproject.toml .
COPY env/ env/
COPY baseline.py .
COPY openenv.yaml .

RUN uv sync --no-dev

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY --from=builder /app .

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
