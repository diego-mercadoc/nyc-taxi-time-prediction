import mlflow

model_name = "nyc-taxi-model"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

requirements_file = "model_requirements.txt"

# Obtén las dependencias del modelo en formato 'requirements'
dependencies = mlflow.pyfunc.get_model_dependencies(
    model_uri=model_uri,
    format="requirements"
)

# Escribe las dependencias en el archivo especificado
with open(requirements_file, "w") as f:
    for dep in dependencies:
        f.write(f"{dep}\n")

print(f"Dependencias guardadas en {requirements_file}")
