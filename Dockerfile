FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY ./stellar-go/frontend/package.json ./stellar-go/frontend/package-lock.json ./
RUN npm install
COPY ./stellar-go/frontend/ ./
RUN npm run build

FROM golang:1.24 AS stellar-builder

RUN apt update && \
    apt install -y pciutils \
    gcc libgl1-mesa-dev xorg-dev libxkbcommon-dev

WORKDIR /app
COPY ./stellar-go/go.mod ./stellar-go/go.sum ./
RUN go mod download

COPY ./stellar-go/Makefile ./Makefile
COPY ./stellar-go/p2p ./p2p
COPY ./stellar-go/core ./core
COPY ./stellar-go/cmd ./cmd
COPY ./stellar-go/assets.go ./assets.go
COPY ./stellar-go/stellar-client ./stellar-client
COPY --from=frontend-builder /app/dist ./frontend/dist
COPY ./stellar-go/frontend/assets.go ./frontend/assets.go
RUN CGO_ENABLED=1 GOOS=linux make build

FROM python:3.11-slim 

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    jq \
    supervisor \
    pciutils \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/log/supervisor

RUN pip install build

COPY ./stefan-fl/ /stefan-fl
RUN cd /stefan-fl  \
    && python -m build \
    && WHL_PATH=$(find dist -name "*.whl" | head -n 1) \
    && mkdir -p /dist && mv $WHL_PATH /dist/ && WHL_PATH=/dist/$(basename $WHL_PATH) \
    && pip install --no-cache-dir $WHL_PATH[openfl] \
    && pip install --no-cache-dir $WHL_PATH[flower,nvflare] \
    && rm -rf /stefan-fl
ENV STEFAN_WHL_DIR=/dist

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./stellar-go/stellar-client /stellar-client
RUN cd /stellar-client \
    && python -m build \
    && pip install dist/*.whl \
    && rm -rf /stellar-client

COPY --from=stellar-builder /app/build/stellar /usr/bin/stellar

COPY ./stellar/*.py .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9417"]