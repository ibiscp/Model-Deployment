from random import randint
from mlflow.pyfunc import PythonModel
import mlflow
import os

class MyModel(PythonModel):

    def __init__(self, seed:int = 100) -> None:
        super().__init__()
        self.seed = seed

    def predict(self, context, model_input):
        return f"{model_input} {self.seed}"


if __name__ == "__main__":

    project_id = 1
    # mlflow.set_tracking_uri("http://localhost:5001")
    # mlflow.set_tracking_uri(os.environ.get('tracking_uri'))
    # os.environ["AWS_ACCESS_KEY_ID"] = "onetask"
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "JRZtI0SLsEDb3imTy03R"
    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    experiment_name = f"project_{project_id}"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    seed_value = randint(0, 1000)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"model_{seed_value}"):
        
        mlflow.log_param("seed", seed_value)
        model = MyModel(seed=seed_value)

        mlflow.pyfunc.log_model(f"model_{seed_value}", python_model=model)
