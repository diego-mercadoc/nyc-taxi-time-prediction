import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import xgboost as xgb
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import task, flow  # Importar task y flow desde prefect PARA QUE JALE EL CODIGO
from mlflow.tracking import MlflowClient  # Importar mlflowclient para hablar con mlefloe
import matplotlib
matplotlib.use('Agg')  # Para que no se abra la ventana de matplotlib

# Activar el entorno virtual
# .\.venv\Scripts\Activate
# prefect server start

# Lw wa poner N_JOBS = -1 para que CORRA COMO BESTIA
N_JOBS = -1

@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
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
    """Add features to the model"""
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

@task(name="Hyper-parameter Tuning")
def hyper_parameter_tuning(X_train, X_val, y_train, y_val, dv):
    mlflow.xgboost.autolog()
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    
    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "xgboost")
            
            # Ponerle N_JOBS = -1 para que corra como bestia (X2 xd) EL ENTRENAMIENTO
            params["n_jobs"] = N_JOBS

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10
            )
            
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            
            mlflow.log_metric("rmse", rmse)
    
        return {'loss': rmse, 'status': STATUS_OK}

    with mlflow.start_run(run_name="Xgboost Hyper-parameter Optimization", nested=True):
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:squarederror',
            'seed': 42,
            'n_jobs': N_JOBS # Ponerle N_JOBS = -1 para LA OPTIMIZACION DE HypParams, que corra como bestia (X2 xd)
        }
        
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["seed"] = 42
        best_params["objective"] = "reg:squarederror"
        best_params["n_jobs"] = N_JOBS # N_JOBS = -1 a LOS MEJORES PARAMETROS
        
        mlflow.log_params(best_params)

    return best_params

@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params):
    with mlflow.start_run(run_name="Best model ever"):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Loguear el modelo entrenado apa registralo luegoo
        mlflow.xgboost.log_model(booster, artifact_path="model") 

    return None

# La nuevaa tarea para registrar el best model MODEL REGSSTRY
@task(name="Register Best Model")
def register_best_model():
    client = MlflowClient()  # Craear el cliente para hablar con mlefloew

    experiment_name = "nyc-taxi-experiment-prefect"
    experiment = client.get_experiment_by_name(experiment_name)

    # Buscar el run con el RMSE mass bajo
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    best_run = runs[0]
    run_id = best_run.info.run_id  # Sacart el run_id automaticamente COMO EN LA TAREA

    model_uri = f"runs:/{run_id}/model"
    model_name = "nyc-taxi-model-prefect"  # Nombre del modelo en el model registry

    # Registrar EL MODELOO MODEL REGSITRY
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"Asignando el alias 'champion' al modelo '{model_name}', versión '{registered_model.version}'")

    # Asignar el alias '@champion' a la versión registrada del modelo
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered_model.version
    )

@flow(name="Main Flow")
def main_flow(year: str, month_train: str, month_val: str):
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    dagshub.init(url="https://dagshub.com/diego-mercadoc/nyc-taxi-time-prediction", mlflow=True)
    
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    best_params = hyper_parameter_tuning(X_train, X_val, y_train, y_val, dv)
    
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)
    
    register_best_model()  # Aqui le Hablamso a la NEW TASK para registrar el BEST MODE;L

main_flow("2024", "01", "02")
