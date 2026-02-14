# ============================================================================
# AACO Dockerfile - AMD AI Compute Observatory
# ============================================================================
# Multi-stage build for optimized production image
# © 2026 Sudheer Ibrahim Daniel Devu
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY pyproject.toml README.md ./
COPY aaco/ ./aaco/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim as production

LABEL maintainer="Sudheer Ibrahim Daniel Devu"
LABEL description="AMD AI Compute Observatory - Performance Science Engine"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/SID-Devu/AMD-AI-Compute-Observatory"
LABEL org.opencontainers.image.licenses="Proprietary"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r aaco && useradd -r -g aaco aaco

# Copy built wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Copy configuration files
COPY configs/ ./configs/

# Set ownership
RUN chown -R aaco:aaco /app

# Switch to non-root user
USER aaco

# Default environment variables
ENV AACO_CONFIG_DIR=/app/configs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import aaco; print('healthy')" || exit 1

# Default command
ENTRYPOINT ["aaco"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# Stage 3: Development Image
# -----------------------------------------------------------------------------
FROM python:3.10-slim as development

LABEL maintainer="Sudheer Ibrahim Daniel Devu"
LABEL description="AMD AI Compute Observatory - Development Image"

WORKDIR /workspace

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install in development mode with all extras
RUN pip install --no-cache-dir -e ".[all]"

# Install additional dev tools
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    mkdocs \
    mkdocs-material

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["bash"]

# -----------------------------------------------------------------------------
# Stage 4: Dashboard Image
# -----------------------------------------------------------------------------
FROM production as dashboard

USER root

# Install dashboard dependencies
RUN pip install --no-cache-dir \
    streamlit>=1.28.0 \
    plotly>=5.18.0

USER aaco

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "aaco/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
