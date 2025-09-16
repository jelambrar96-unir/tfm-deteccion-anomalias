import json
import mlflow
from mlflow.tracking import MlflowClient

from core import MLFLOW_EXPERIMENT_NAME, MLFLOW_SERVER, MLFLOW_TRACKING_URI


mlflow.set_tracking_uri(MLFLOW_SERVER)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)




def load_model_from_mlflow(run_name: str):

    experiment = mlflow.search_experiments(filter_string=f"name='{MLFLOW_EXPERIMENT_NAME}'")[0]
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["metrics.roc_auc DESC"]
    )

    # print(runs.to_dict(orient='records'))
    #  print(type(runs))
    if runs.shape[0] == 0:
        raise ValueError(f"No se encontró ningún run con el nombre: {run_name}")


    run_id = runs.iloc[0]['run_id']
    # print(run_id)
    model_uri = f"runs:/{run_id}/model"

    # Cargar el modelo
    model = mlflow.sklearn.load_model(model_uri)
    return model


