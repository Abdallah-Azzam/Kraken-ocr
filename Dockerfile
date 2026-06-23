FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV DEFAULT_LANGUAGE=en
ENV DEFAULT_DOCUMENT_TYPE=printed
ENV DEFAULT_BATCH_SIZE=32
ENV DEFAULT_NUM_LINE_WORKERS=4
ENV DEFAULT_BINARIZE=false
ENV DEFAULT_INCLUDE_LINES=true
ENV KRAKEN_PRECISION=bf16-mixed
ENV XDG_DATA_HOME=/models/data
ENV HOME=/root

WORKDIR /

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash libgl1 libx11-6 \
        software-properties-common build-essential -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install python3.11 python3.11-dev python3.11-venv python3-pip -y --no-install-recommends && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r /requirements.txt --no-cache-dir

COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && rm /fetch_models.py

COPY src .
COPY handler.py .
COPY assets /assets
COPY test_input.json .

CMD python -u /rp_handler.py
