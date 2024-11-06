# model/main.py

import pickle
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow import MlflowClient
import os

# MLflow settings
dagshub_repo = "https://dagshub.com/diego-mercadoc/nyc-taxi-time-prediction"
# dagshub_repo = "https://dagshub.com/Pepe-Chuy/nyc-taxi-time-prediction"

# MLFLOW_TRACKING_URI: URI de seguimiento de MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/diego-mercadoc/nyc-taxi-time-prediction.mlflow"

# Configurar MLflow
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Buscar el run con el menor RMSE en el experimento especificado
runs = mlflow.search_runs(
    order_by=['metrics.rmse ASC'],
    output_format="list",
    experiment_names=["nyc-taxi-experiment-prefect"]
)

if not runs:
    raise ValueError("No runs found in experiment 'nyc-taxi-experiment-prefect'.")

run_ = runs[0]
run_id = run_.info.run_id

# Descargar artifacts del preprocesador
client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

# Cargar el preprocesador
with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

# Configuración del modelo en MLflow
model_name = "nyc-taxi-model"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

# Cargar el modelo campeón
champion_model = mlflow.pyfunc.load_model(model_uri=model_uri)

def preprocess(input_data):
    """
    Preprocesa los datos de entrada para el modelo.
    """
    input_dict = {
        'PU_DO': f"{input_data.PULocationID}_{input_data.DOLocationID}",
        'trip_distance': input_data.trip_distance,
    }
    return dv.transform(input_dict)

def predict(input_data):
    """
    Realiza la predicción utilizando el modelo cargado.
    """
    X_pred = preprocess(input_data)
    return champion_model.predict(X_pred)

# Inicializar la aplicación FastAPI
app = FastAPI()

class InputData(BaseModel):
    """
    Modelo de datos para la solicitud de predicción.
    """
    PULocationID: str
    DOLocationID: str
    trip_distance: float

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    """
    Endpoint para realizar predicciones de tiempo de viaje.
    """
    try:
        prediction = predict(input_data)[0]
        return {
            "prediction": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar la aplicación con Uvicorn si se ejecuta directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)