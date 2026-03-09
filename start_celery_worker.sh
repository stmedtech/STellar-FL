#!/bin/bash

# Start Celery worker for Stellar task execution server
# Make sure Redis is running before starting the worker

echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=info --concurrency=2 $@
