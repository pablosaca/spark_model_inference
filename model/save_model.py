import os
import json

from typing import Optional, List, Literal, Dict

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from onnxmltools.convert.sparkml.convert import convert
from onnxmltools.convert.common.data_types import FloatTensorType, StringTensorType
from onnxmltools.utils import save_model

from model.data import get_data_path
from utils.utils import get_logger

logger = get_logger()


default_types = [
    ('building_type', StringTensorType([None, 1])),
    ('object_type', StringTensorType([None, 1])),
    ('rooms_label', StringTensorType([None, 1])),
    ('level_group', StringTensorType([None, 1])),
    ('levels_group', StringTensorType([None, 1])),
    ('open_kitchen', StringTensorType([None, 1])),
    ('kitchen_area', FloatTensorType([None, 1]))
]


def save_artifact(
        spark: SparkSession,
        model: PipelineModel,
        feats_types: Optional[List[tuple]] = None,
        folder_name: str = "artifact",
        model_name: Literal["modelo_precios", "modelo_precios_1", "modelo_precios_2"] = "modelo_precios"
) -> None:
    """
    Guardado del modelo. De spark a onnx
    """

    initial_types = default_types if feats_types is None else feats_types

    # 'model' es el PipelineModel entrenado
    onnx_model = convert(model, name="REPSparkModel", initial_types=initial_types, spark_session=spark)
    logger.info("Conversión del modelo de spark a onnx")

    model_name = f"{model_name}.onnx"
    model_path = get_data_path(folder_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_path = str(model_path / model_name)
    save_model(onnx_model, file_path)
    logger.info(f"Modelo guardado: {file_path}")


def save_metrics(
    metrics: Dict[str, float],
    folder_name: str = "metrics",
    file_name: str = "metrics.json"
) -> None:
    """
    Guarda un diccionario de métricas en formato json
    """
    metrics_ = {"rmse": metrics}

    model_path = get_data_path(folder_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_path = str(model_path / file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics_, f, indent=4, ensure_ascii=False)

    logger.info(f"Métricas guardadas en: {file_path}")
