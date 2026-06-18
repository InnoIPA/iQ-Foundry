FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    bash \
    adb \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/iQ-Foundry

COPY requirements/host.txt /tmp/host.txt

RUN python3 -m venv /opt/iq-foundry-venv && \
    /opt/iq-foundry-venv/bin/pip install --upgrade pip && \
    /opt/iq-foundry-venv/bin/pip install --no-cache-dir -r /tmp/host.txt

ENV PATH="/opt/iq-foundry-venv/bin:${PATH}"

CMD ["bash"]