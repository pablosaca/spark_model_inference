from typing import Dict, Literal, Tuple
from pyspark.sql import DataFrame as DF

from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from utils.utils import get_logger

logger = get_logger()


def model_train(df: DF) -> PipelineModel:

    # Pipeline de preprocesamiento
    indexer = StringIndexer(
        inputCols=[
            "building_type", "object_type",
            "rooms_label", "level_group", "levels_group", "open_kitchen"
            ],
        outputCols=[
            "building_type_idx", "object_type_idx",
            "rooms_label_idx", "level_group_idx", "levels_group_idx", "open_kitchen_idx"
        ],
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=[
            "building_type_idx", "object_type_idx",
            "rooms_label_idx", "level_group_idx", "levels_group_idx", "open_kitchen_idx"
        ],
        outputCols=[
            "building_type_vec", "object_type_vec",
            "rooms_label_vec", "level_group_vec", "levels_group_vec", "open_kitchen_vec"
        ],
        handleInvalid="keep",
        dropLast=True
    )

    # VectorAssembler para combinar todas las features
    assembler = VectorAssembler(
        inputCols=[
            "building_type_vec", "object_type_vec",
            "rooms_label_vec", "level_group_vec", "levels_group_vec", "open_kitchen_vec",
            "kitchen_area"
        ],
        outputCol="features"
    )

    # Modelo de regresión lineal
    lr = LinearRegression(
        featuresCol="features", labelCol="target", elasticNetParam=0.25, maxIter=100, regParam=0.07, solver="normal"
    )

    # Creación pipeline
    pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])

    # Entrenamiento del modelo
    fitted_model = pipeline.fit(df)
    logger.info("Entrenado el modelo")
    return fitted_model


def model_evaluation(
        model: LinearRegression,
        df: DF,
        sample_name: Literal["train", "test"] = "train",
        metric_name: Literal["rmse"] = "rmse",
        label_col: Literal["target", "price"] = "target"
) -> Dict[str, float]:

    prediction_df = model.transform(df)
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName=metric_name)
    value = evaluator.evaluate(prediction_df)
    logger.info(f"Evaluación del modelo para la muestra {sample_name}")
    return {sample_name: value}


def train_test_split(df: DF, split_value: float = 0.7) -> Tuple[DF, DF]:
    train_df, test_df = df.randomSplit([split_value, 1 - split_value])
    logger.info(f"División muestra entrenamiento y validación. Uso de un split de {split_value}")
    return train_df, test_df
