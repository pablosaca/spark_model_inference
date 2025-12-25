import os
import sys

from pyspark.sql import SparkSession

from utils.utils import get_logger

logger = get_logger()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 2. ELIMINA SPARK_HOME temporalmente para evitar conflictos de versiones JAR
if 'SPARK_HOME' in os.environ:
    del os.environ['SPARK_HOME']


# creamos una sesión de spark
def spark_session() -> SparkSession:

    spark = (
        SparkSession.builder
        .master("local[*]")  # IMPORTANTE: Forzar modo local
        .appName("Modelización spark local")
     #   .config("spark.jars.packages", "ml.combust.mleap:mleap-spark_2.12:0.23.0")
        .config("spark.driver.host", "127.0.0.1")  # identificamos la IP local
        .config("spark.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "8g")  # fijamos memoria en driver y executor por si se nos peta
        .config("spark.executor.memory", "8g")
        # PARÁMETROS CRÍTICOS PARA JAVA 17
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED " +
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " +
                "--add-opens=java.base/java.lang=ALL-UNNAMED " +
                "--add-opens=java.base/java.util=ALL-UNNAMED " +
                "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED")  # Nuevo flag
        .getOrCreate()
    )
    logger.info("Inciada sesión de spark")
    return spark
