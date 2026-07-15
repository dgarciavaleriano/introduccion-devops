# Adult Income ML — Pipeline MLOps con GitHub Actions y Azure

Proyecto académico de **introducción a DevOps y MLOps** que implementa el ciclo de vida completo de un modelo de aprendizaje automático: descarga y preparación de datos, entrenamiento, evaluación, registro en MLflow/Azure Machine Learning, pruebas automatizadas, construcción de una imagen Docker y despliegue de una API de inferencia en Azure Container Apps.

El modelo utiliza un **Random Forest Classifier** para predecir si los ingresos anuales de una persona superan los 50.000 USD a partir del conjunto de datos **Adult / Census Income** de UCI.

## Características principales

- Entrenamiento de un modelo `RandomForestClassifier` con Scikit-learn.
- Preprocesamiento de variables numéricas y categóricas.
- Registro automático de métricas y artefactos con MLflow.
- Registro y promoción del modelo al entorno `Production`.
- Pruebas unitarias y pruebas específicas del modelo con Pytest.
- Generación de informes de cobertura.
- API REST de inferencia desarrollada con FastAPI.
- Empaquetado de la API mediante Docker.
- Publicación de imágenes en GitHub Container Registry.
- Despliegue automatizado en Azure Container Apps.
- Pipelines de integración, entrenamiento y despliegue con GitHub Actions.

## Arquitectura del proyecto

```text
introduccion-devops/
├── .github/
│   └── workflows/
│       ├── build.yml          # Entrenamiento, pruebas y registro del modelo
│       ├── integration.yml    # Pruebas unitarias y cobertura en Pull Requests
│       └── deploy.yml         # Construcción Docker y despliegue en Azure
├── deployment/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py            # API FastAPI de inferencia
│   └── Dockerfile
├── model_tests/
│   └── test_model.py          # Validación del modelo entrenado
├── scripts/
│   └── register_model.py      # Registro y promoción del modelo en MLflow
├── src/
│   ├── data_loader.py         # Carga y preprocesamiento de datos
│   ├── evaluate.py            # Evaluación del modelo
│   ├── main.py                # Ejecución del entrenamiento
│   └── model.py               # Definición y entrenamiento del modelo
├── unit_tests/
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   └── test_model.py
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Flujo MLOps

```text
Pull Request
    │
    ▼
Integration workflow
    ├── Instalación de dependencias
    ├── Ejecución de pruebas unitarias
    ├── Cálculo de cobertura
    └── Publicación del resultado en el Pull Request

Push a main / ejecución manual
    │
    ▼
Build Model workflow
    ├── Descarga del dataset Adult
    ├── Entrenamiento del Random Forest
    ├── Registro de métricas y artefactos en MLflow
    ├── Pruebas del modelo
    └── Registro y promoción del modelo a Production

Ejecución manual
    │
    ▼
Deploy Model workflow
    ├── Construcción de la imagen Docker
    ├── Publicación en GitHub Container Registry
    ├── Despliegue en Azure Container Apps
    ├── Comprobación del endpoint /health
    └── Publicación de la URL de Swagger
```

## Tecnologías utilizadas

- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- Joblib
- MLflow
- Azure Machine Learning
- Azure Container Apps
- Azure CLI
- FastAPI
- Uvicorn
- Docker
- GitHub Actions
- GitHub Container Registry
- Pytest y Pytest-Cov

## Requisitos previos

Para ejecutar el proyecto en local se necesita:

- Python 3.10 o compatible.
- Git.
- Docker, únicamente para ejecutar la API en contenedor.
- Acceso a un servidor MLflow o a un workspace de Azure Machine Learning.
- Credenciales de Azure para los procesos de registro y despliegue.

## Instalación local

Clonar el repositorio:

```bash
git clone https://github.com/dgarciavaleriano/introduccion-devops.git
cd introduccion-devops
```

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

El proyecto utiliza el dataset público **Adult / Census Income** de UCI Machine Learning Repository.

El workflow de entrenamiento descarga automáticamente los ficheros necesarios. Para ejecutar el entrenamiento en local, hay que crear primero la carpeta de datos y descargar ambos archivos:

```bash
mkdir -p data/raw
curl -o data/raw/adult.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -o data/raw/adult.test \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

En Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path data/raw
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" -OutFile "data/raw/adult.data"
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" -OutFile "data/raw/adult.test"
```

## Variables de entorno para el entrenamiento

El script `src/main.py` utiliza las siguientes variables:

| Variable | Descripción | Obligatoria |
|---|---|---:|
| `MLFLOW_TRACKING_URL` | URI del servidor MLflow o workspace de Azure ML | Sí |
| `EXPERIMENT_NAME` | Nombre del experimento de MLflow | Sí |
| `RUN_NAME` | Nombre asignado a la ejecución | No |
| `MODEL_NAME` | Nombre con el que se registra el modelo | Para el registro |

Ejemplo:

```bash
export MLFLOW_TRACKING_URL="<URI_DE_MLFLOW>"
export EXPERIMENT_NAME="adult-income-model"
export MODEL_NAME="adult-income-model"
export RUN_NAME="local-run"
```

## Entrenamiento del modelo

Con el entorno virtual activado, el dataset descargado y las variables configuradas:

```bash
python src/main.py
```

El proceso realiza las siguientes operaciones:

1. Carga los conjuntos de entrenamiento y prueba.
2. Elimina registros con valores ausentes.
3. Codifica las variables categóricas con `LabelEncoder`.
4. Normaliza las características mediante `StandardScaler`.
5. Entrena un `RandomForestClassifier`.
6. Calcula la precisión y el informe de clasificación.
7. Guarda el modelo y los objetos de preprocesamiento.
8. Registra parámetros, métricas y artefactos en MLflow.
9. Guarda el identificador de ejecución en `run_id.txt`.

Los artefactos locales se generan en:

```text
models/
├── model.pkl
├── scaler.pkl
└── encoders.pkl
```

## Ejecución de pruebas

### Pruebas unitarias

```bash
PYTHONPATH=src pytest unit_tests/ -v
```

### Pruebas con cobertura

```bash
PYTHONPATH=src pytest unit_tests/ \
  --cov=model \
  --cov=evaluate \
  --cov=data_loader \
  --cov-report=term \
  --cov-report=html
```

El informe HTML se genera en:

```text
htmlcov/index.html
```

### Pruebas del modelo entrenado

Estas pruebas requieren que el entrenamiento se haya ejecutado previamente y que existan los artefactos dentro de `models/`:

```bash
pytest model_tests/test_model.py -v
```

Las pruebas comprueban:

- Que el modelo pueda cargarse correctamente.
- Que el tamaño de las predicciones sea el esperado.
- Que las clases predichas sean `0` o `1`.
- Que la precisión del modelo sea igual o superior al 80 %.

## Registro del modelo

Después del entrenamiento, el modelo puede registrarse mediante:

```bash
export RUN_ID="$(cat run_id.txt)"
export MODEL_NAME="adult-income-model"
python scripts/register_model.py
```

El script registra la versión del modelo en MLflow y la promociona al estado `Production`.

## API de inferencia

La API está implementada con FastAPI y expone los siguientes endpoints:

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Comprueba el estado de la aplicación y del modelo |
| `POST` | `/predict` | Genera una predicción de ingresos |
| `GET` | `/metrics` | Devuelve el número total de predicciones |
| `GET` | `/docs` | Interfaz Swagger/OpenAPI |

### Variables de entorno de la API

| Variable | Descripción |
|---|---|
| `MLFLOW_TRACKING_URL` | URI del servidor MLflow o Azure ML |
| `MODEL_URI` | URI del modelo registrado, por ejemplo `models:/adult-income-model/Production` |
| `AZURE_TENANT_ID` | Identificador del tenant de Azure |
| `AZURE_CLIENT_ID` | Identificador de la aplicación o service principal |
| `AZURE_CLIENT_SECRET` | Secreto del service principal |

### Ejecutar la API en local

```bash
export MLFLOW_TRACKING_URL="<URI_DE_MLFLOW>"
export MODEL_URI="models:/adult-income-model/Production"
export AZURE_TENANT_ID="<AZURE_TENANT_ID>"
export AZURE_CLIENT_ID="<AZURE_CLIENT_ID>"
export AZURE_CLIENT_SECRET="<AZURE_CLIENT_SECRET>"

uvicorn deployment.app.main:app --host 0.0.0.0 --port 8080
```

Swagger estará disponible en:

```text
http://localhost:8080/docs
```

## Ejemplo de predicción

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'
```

Respuesta esperada:

```json
{
  "prediction": [0]
}
```

Interpretación de las clases:

- `0`: ingresos anuales inferiores o iguales a 50.000 USD.
- `1`: ingresos anuales superiores a 50.000 USD.

## Uso con Docker

Construir la imagen desde la raíz del repositorio:

```bash
docker build -t adult-income-api -f deployment/Dockerfile .
```

Ejecutar el contenedor:

```bash
docker run --rm -p 8080:8080 \
  -e MLFLOW_TRACKING_URL="<URI_DE_MLFLOW>" \
  -e MODEL_URI="models:/adult-income-model/Production" \
  -e AZURE_TENANT_ID="<AZURE_TENANT_ID>" \
  -e AZURE_CLIENT_ID="<AZURE_CLIENT_ID>" \
  -e AZURE_CLIENT_SECRET="<AZURE_CLIENT_SECRET>" \
  adult-income-api
```

Comprobar el estado:

```bash
curl http://localhost:8080/health
```

## GitHub Actions

El repositorio contiene tres workflows.

### 1. Integration

Archivo: `.github/workflows/integration.yml`

Se ejecuta automáticamente en cada Pull Request y también puede iniciarse manualmente. Sus funciones principales son:

- Instalar las dependencias.
- Ejecutar las pruebas unitarias.
- Calcular la cobertura.
- Generar resultados JUnit y HTML.
- Publicar un comentario con los resultados en el Pull Request.
- Marcar el job como fallido cuando las pruebas no se superan.

### 2. Build Model

Archivo: `.github/workflows/build.yml`

Se ejecuta al hacer push sobre `main` o manualmente desde GitHub Actions. Realiza:

- Autenticación en Azure.
- Descarga del dataset.
- Entrenamiento del modelo.
- Registro de la ejecución en MLflow.
- Ejecución de las pruebas del modelo.
- Registro y promoción del modelo a `Production`.

### 3. Deploy Model

Archivo: `.github/workflows/deploy.yml`

Se ejecuta manualmente y realiza:

- Inicio de sesión en GitHub Container Registry.
- Construcción y publicación de la imagen Docker.
- Autenticación en Azure.
- Actualización de la aplicación en Azure Container Apps.
- Configuración de las variables de entorno de producción.
- Comprobación automática del endpoint `/health`.
- Presentación de la URL final de Swagger.

## Configuración de GitHub

### Secrets requeridos

En `Settings > Secrets and variables > Actions > Secrets`:

| Secret | Uso |
|---|---|
| `AZURE_CREDENTIALS` | Credenciales JSON del service principal de Azure |
| `MLFLOW_TRACKING_URL` | URI privada de MLflow/Azure Machine Learning |
| `GH_PAT` | Token para publicar y descargar imágenes en GHCR |

Ejemplo de estructura de `AZURE_CREDENTIALS`:

```json
{
  "clientId": "<CLIENT_ID>",
  "clientSecret": "<CLIENT_SECRET>",
  "subscriptionId": "<SUBSCRIPTION_ID>",
  "tenantId": "<TENANT_ID>"
}
```

> Nunca se deben incluir credenciales reales, tokens o secretos directamente en el repositorio.

### Variables requeridas

En `Settings > Secrets and variables > Actions > Variables`:

| Variable | Valor recomendado |
|---|---|
| `EXPERIMENT_NAME` | `adult-income-model` |
| `MODEL_NAME` | `adult-income-model` |

## Salud y métricas

Ejemplo de respuesta de `/health` cuando el modelo está preparado:

```json
{
  "status": "ok",
  "worker_state": "ready",
  "model_loaded": true
}
```

Ejemplo de respuesta de `/metrics`:

```text
total_predictions 12
```

## Buenas prácticas aplicadas

- Separación entre carga de datos, modelo, evaluación y despliegue.
- Uso de variables de entorno y secretos de GitHub.
- Integración continua en Pull Requests.
- Validación automática de la precisión mínima.
- Registro versionado del modelo.
- Contenerización reproducible.
- Comprobación automática del despliegue.
- Exposición de endpoints de salud y métricas.

## Posibles mejoras

- Sustituir `LabelEncoder` por un pipeline de Scikit-learn que gestione categorías desconocidas.
- Añadir validaciones más restrictivas a los campos de entrada de la API.
- Incorporar autenticación y autorización en el endpoint de predicción.
- Publicar métricas en formato Prometheus.
- Añadir pruebas de integración para la API.
- Fijar versiones exactas de las acciones de GitHub y dependencias.
- Utilizar alias de MLflow en lugar de etapas si la versión desplegada lo recomienda.
- Añadir análisis estático, formateo y comprobaciones de seguridad.
- Gestionar la infraestructura de Azure mediante Terraform o Bicep.

## Equipo

Proyecto desarrollado por el **Grupo 4** como parte de la práctica de introducción a DevOps/MLOps.

## Licencia

Este repositorio tiene finalidad académica y educativa. El dataset Adult pertenece a sus respectivos autores y se distribuye desde UCI Machine Learning Repository.
