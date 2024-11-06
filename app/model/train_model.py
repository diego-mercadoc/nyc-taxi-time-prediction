# model/train_model.py

import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

# Simulación de datos de ejemplo (reemplaza esto con tus datos reales)
data = {
    'PULocationID': [236, 65, 74, 33, 166, 226, 7, 42],
    'DOLocationID': [239, 170, 262, 209, 239, 226, 129, 75],
    'trip_distance': [1.98, 6.54, 3.08, 2.00, 2.01, 0.31, 2.32, 2.69],
    'trip_time': [15, 35, 20, 18, 19, 12, 17, 19]  # Variable objetivo
}

df = pd.DataFrame(data)

# Preprocesamiento
dv = DictVectorizer()
X = dv.fit_transform(df[['PULocationID', 'DOLocationID', 'trip_distance']].to_dict(orient='records'))
y = df['trip_time'].values

# Entrenamiento del modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluación del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE en el conjunto de prueba: {rmse}")

# Guardar el preprocesador
os.makedirs('preprocessor', exist_ok=True)
with open('preprocessor/preprocessor.b', 'wb') as f:
    pickle.dump(dv, f)

# Guardar el modelo
os.makedirs('models', exist_ok=True)
with open('models/nyc_taxi_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo y preprocesador guardados exitosamente.")
