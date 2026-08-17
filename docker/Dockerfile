# Multi-stage build keeps the runtime image small and free of build tools.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
COPY broker ./broker
COPY strategies ./strategies
COPY ai ./ai
COPY risk ./risk
COPY portfolio ./portfolio
COPY execution ./execution
COPY backtesting ./backtesting
COPY data ./data
COPY monitoring ./monitoring
COPY database ./database
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
# Run as a non-root user: the agent has no need for root, and a trading
# process with root in a container is an unnecessary blast radius.
RUN useradd --create-home --shell /bin/bash trader
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=trader:trader . /app
USER trader

# TRADING_MODE is deliberately NOT set here. An unset mode falls back to
# the PAPER default in config.py. Baking a mode into the image is how a
# container ends up trading live because someone reused a tag.
ENV PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); \
    sys.exit(0 if r.status==200 else 1)" || exit 1

EXPOSE 8000
CMD ["python", "-m", "trading_agent", "status"]
