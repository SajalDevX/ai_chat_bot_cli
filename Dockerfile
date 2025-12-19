FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "-m", "ai_chat_bot.main"]