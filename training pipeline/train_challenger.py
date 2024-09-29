import matplotlib
matplotlib.use('Agg')  # Para que no se abra la ventana de matplotlib

import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import xgboost as xgb
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import task, flow
from mlflow.tracking import MlflowClient

# Activar el entorno virtual
# .\.venv\Scripts\Activate
# prefect server start

N_JOBS = -1

@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Leer datos y preparar DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Agregar características al modelo"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="Train Model")
def train_model(X_train, X_val, y_train, y_val, dv, params, model_name):
    with mlflow.start_run(run_name=model_name):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.xgboost.log_model(booster, artifact_path="model")

    return rmse

@flow(name="Challenger Flow")
def challenger_flow():
    dagshub.init(url="https://dagshub.com/diego-mercadoc/nyc-taxi-time-prediction", mlflow=True)

    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    df_train = read_data("../data/green_tripdata_2024-01.parquet")
    df_val = read_data("../data/green_tripdata_2024-02.parquet")

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Entrenar primer modelo
    params1 = {
        'max_depth': 10,
        'learning_rate': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'objective': 'reg:squarederror',
        'seed': 42,
        'n_jobs': N_JOBS
    }
    rmse1 = train_model(X_train, X_val, y_train, y_val, dv, params1, "Model 1")

    # Entrenar segundo modelo
    params2 = {
        'max_depth': 20,
        'learning_rate': 0.05,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'objective': 'reg:squarederror',
        'seed': 42,
        'n_jobs': N_JOBS
    }
    rmse2 = train_model(X_train, X_val, y_train, y_val, dv, params2, "Model 2")

    # Decidir cuál modelo es mejor
    if rmse1 < rmse2:
        best_rmse = rmse1
        best_model_name = "Model 1"
    else:
        best_rmse = rmse2
        best_model_name = "Model 2"

    # Registrar el modelo como 'challenger'
    client = MlflowClient()
    experiment = client.get_experiment_by_name("nyc-taxi-experiment-prefect")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{best_model_name}'",
        max_results=1
    )

    best_run = runs[0]
    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    model_name = "nyc-taxi-model-prefect"

    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Asignar alias 'challenger'
    client.set_registered_model_alias(
        name=model_name,
        alias="challenger",
        version=registered_model.version
    )

    # Obtener el 'champion' actual
    versions = client.get_latest_versions(name=model_name, stages=None)
    champion_version = None
    for v in versions:
        if "champion" in v.aliases:
            champion_version = v
            break

    # Comparar desempeño y actualizar alias
    if champion_version:
        champion_rmse = float(champion_version.tags.get("rmse", float('inf')))
        if best_rmse < champion_rmse:
            # Actualizar alias
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=registered_model.version
            )
            print(f"El nuevo 'champion' es la versión {registered_model.version}")
        else:
            print(f"El 'champion' actual se mantiene en la versión {champion_version.version}")
    else:
        # Si no hay 'champion', asignar el actual
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=registered_model.version
        )
        print(f"El 'champion' es la versión {registered_model.version}")

if __name__ == "__main__":
    challenger_flow()