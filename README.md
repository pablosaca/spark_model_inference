
# Modelo de Predicción de Precio de Vivienda en Rusia con PySpark + Inferencia con FastAPI

Este proyecto implementa un modelo de regresión lineal múltiple en PySpark 
para predecir el precio de vivienda en Rusia y prepara el modelo para inferencia y despliegue 
con FastAPI, exportándolo a formato ONNX.

Los datos han sido extraídos del siguiente enlace: https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021

Nota: para trabajar correctamente es necesario crearse una carpeta `data` en el proyecto y 
disponer dentro el fichero de datos con el nombre `input_data.csv`

## Flujo de trabajo
### 1. Entrenamiento del modelo

Se crea un Pipeline de PySpark que incluye:

- Preprocesamiento de variables categóricas (StringIndexer + OneHotEncoder)
- VectorAssembler para unir features numéricas y codificadas
- Modelo de LinearRegression (entrenamiento con el dataset de viviendas en Rusia)
- Evaluación el modelo (métrica RMSE sobre train y test)

### 2. Preparación del modelo para inferencia

- Se convierte el modelo entrenado a ONNX para poder usarlo en FastAPI
- Se utiliza onnxmltools para la conversión y onnxruntime para la inferencia

### 3. Inferencia

- Se define un endpoint /spark_model que recibe un JSON con las features del piso:

```
{
  "area": 42.5,
  "kitchen_area": 6.5,
  "building_type": 4,
  "object_type": 0,
  "rooms": 3,
  "level": 5,
  "levels": 10
}
```

- Se llevan a cabo validaciones de entrada y conversión de inputs para adecuarse al modelo

## Requisitos

Crear el entorno de trabajo con anaconda. 

Por comodidad y, de cara a una simulación del entorno productivo, 
será necesario crear uno para entrenamiento y otro para inferencia:

- Entrenamiento

```
conda create -n spark_model_train python=3.9
```

- Inferencia

```
conda create -n spark_model_predict python=3.9
```

Recuerda que para activar el entorno virtual usa la siguiente instrucción

```
conda activate spark_model_inference
```

Utilizando pip instala las dependencias del fichero xxxx_requirements.txt (con xxxx inference o model)

```
pip install -r xxx_requirements.txt
```

- Entrenamiento

```
pyspark==3.5.0
numpy==2.0.2
onnxmltools==1.14.0
```

- Inferencia

```
numpy==2.0.2
uvicorn==0.39.0
fastapi==0.127.0
onnxruntime==1.19.2
```

