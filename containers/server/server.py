from mlflow.pyfunc import PythonModel
import random
import docker
import os
import requests
import socket

from fastapi import FastAPI, HTTPException

def get_port_number():
    while True:
        port = random.randint(8000, 8500)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('host.docker.internal', port)) != 0:
                return port

class MyModel(PythonModel):

    def __init__(self, seed:int = 100) -> None:
        super().__init__()
        self.seed = seed

    def predict(self, context, model_input):
        return f"{model_input[0]} {self.seed}"


app = FastAPI()
models = []
docker_client = docker.from_env()

@app.post("/load")
def load_model(model: str, project_id: str):
    port = get_port_number()

    container = docker_client.containers.run(
        "serve:latest",
        ports={'8000/tcp': port},
        network="mlops_default",
        environment={
            'model': model,
            'project_id': project_id,
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
        },
        detach=True,
    )

    data = {
        'container_id': container.id,
        'internal_endpoint': f"http://host.docker.internal:{port}",
        'external_endpoint': f"http://localhost:{port}",
    }
    models.append(data)

    return data

@app.post("/train")
def train():

    container = docker_client.containers.run(
        "train:latest",
        network="mlops_default",
        environment= {
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
        },
        detach=True,
        remove=True,
    )

    return {'status': 'ok'}

@app.get("/predict/{item}")
def predict(item:str):
    latest_ready_model = None
    endpoint = None
    for model in models[::-1]:
        endpoint = model['internal_endpoint']
        
        if bool(requests.get(f'{endpoint}/ready').json()['ready']):
            latest_ready_model = endpoint
            break

    if not latest_ready_model:
        raise HTTPException(status_code=500, detail="No model is ready!")

    prediction = requests.get(f"{endpoint}/predict/{item}")
    return {"prediction": prediction.json()['prediction']}

@app.get("/list")
def list():
    return models

@app.on_event("shutdown")
def shutdown():
    print("Shutting down models")
    for model in models:
        container_id = model['container_id']
        container = docker_client.containers.get(container_id).stop()