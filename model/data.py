from typing import Literal
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as DF


def get_data_path(folder_name: str) -> Path:
    """
    Crea el directorio
    """

    # directorio donde está la carpeta de este script y sube un nivel
    base_path = Path(__file__).resolve().parent.parent
    folder_path = base_path / folder_name
    return folder_path


def load_data(
        spark: SparkSession,
        folder_name: str = "data",
        file_name: Literal["input_data"] = "input_data",
        file_format: Literal["csv"] = "csv"
) -> DF:
    """
    Lectura de los datos en csv y los convierte en un dataframe de spark
    """
    data_path = get_data_path(folder_name)
    file_path = str(data_path / f"{file_name}.{file_format}")
    df = spark.read.csv(file_path, sep=";", header=True, inferSchema=True)
    return df
