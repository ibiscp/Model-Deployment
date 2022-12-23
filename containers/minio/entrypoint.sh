#!/bin/sh

# Create the /data/mlflow folder if it doesn't already exist
if [ ! -d "/data/mlflow" ]; then
    mkdir -p /data/mlflow
fi

# Start the Minio server
exec minio server /data --address :9000 --console-address ':9001'
