import os
from threading import Thread
from time import sleep
from fastapi import FastAPI
from uvicorn.main import logger
import mlflow


class Model:
    def __init__(self,
        model_name:str = os.environ.get('model'),
        project_id:str = os.environ.get('project_id')):

        self.model_name = model_name
        self.project_id = project_id

        self.model = None
        self.ready = False

    def load(self):
        logger.info(f"Loading model {self.model_name}")
        # self.model = mlflow.pyfunc.load_model(f'models:/{self.model_name}/{self.stage}')
        # self.model = mlflow.pyfunc.load_model('s3://minio:9000/mlflow/models/MyModel/Production/')
        self.model = mlflow.pyfunc.load_model(f'runs:/{self.project_id}/{self.model_name}')

        sleep(10)

        logger.info(f"Model {self.model_name} loaded")
        
        self.ready = True
        

app = FastAPI()
wrapper = Model()

@app.get("/ready")
def ready():
    return {'ready': wrapper.ready}

@app.get("/predict/{item}")
def predict(item:str):
    prediction = wrapper.model.predict(item)
    return {'prediction': prediction}

t1 = Thread(target=wrapper.load)
t1.start()
