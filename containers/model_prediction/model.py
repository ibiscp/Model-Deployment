import os
from threading import Thread
from time import sleep
from fastapi import FastAPI, Body
from uvicorn.main import logger
import mlflow
import pandas as pd


class Model:
    def __init__(
        self,
        model_name: str = os.environ.get("model"),
        project_id: str = os.environ.get("project_id"),
    ):

        self.model_name = model_name
        self.project_id = project_id

        self.model = None
        self.ready = False

    def load(self):
        logger.info(f"Loading model {self.model_name}")
        # self.model = mlflow.pyfunc.load_model(f'models:/{self.model_name}/{self.stage}')
        # self.model = mlflow.pyfunc.load_model('s3://minio:9000/mlflow/models/MyModel/Production/')

        # mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # local_artifact_dir = "runs:/765090d31e86442893532833ccdacfc4/model_330"
        # pathlib.Path(local_artifact_dir).mkdir(parents=True, exist_ok=True)

        # self.model = mlflow.pyfunc.load_model(model_uri=local_artifact_dir)

        self.model = mlflow.pyfunc.load_model(
            f"runs:/{self.project_id}/{self.model_name}"
        )

        # sleep(10)

        logger.info(f"Model {self.model_name} loaded")

        self.ready = True


app = FastAPI()
wrapper = Model()
# wrapper.load()


@app.get("/ready")
def ready():
    return {"ready": wrapper.ready}


@app.post("/predict")
def predict(item: dict = Body(...)):

    X = pd.DataFrame(item["data"], columns=item["columns"])

    prediction = wrapper.model.predict(X)
    return prediction.tolist()


t1 = Thread(target=wrapper.load)
t1.start()


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
