from typing import List, Optional, Literal, Tuple
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as DF

from utils.utils import get_logger

logger = get_logger()


features = [
    "level",
    "levels",
    "rooms",
    "kitchen_area",
    "building_type",
    "object_type"
]


def preprocessing_data(
        df: DF,
        model_features: Optional[List[str]] = None,
        feat_col: Literal["area"] = "area",
        target_col: Optional[Literal["price"]] = "price",
        price_m2_range: Tuple[int] = (80000, 450000)
) -> DF:
    """
    Preprocesado de datos. Se realizan diferentes etapas como:
    - Filtrado de registros incoherentes
    - Creación de variables y agrupación de niveles para variables categóricas
    - Outliers en target
    """

    feats = select_cols(feat_col, model_features, target_col)

    # FILTRADO DE REGISTROS POR INCONSISTENCIAS

    # eliminación de registros que el área de la cocina es mayor que el área total
    # eliminación de registros que el área es menor que cero
    # eliminación de registros uqe el área de la cocina es menor que cero
    # eliminación piso es mayor que el número de bloques del inmueble

    # CREACIÓN DE NUEVAS VARIABLES
    # categorización de planta del piso y del número de plantas del inmueble
    # identificación cocina integrada en el salón

    df = (
        df.select(*feats)
        .filter(
            (F.col(feat_col) > 10) &  # área mayor que 10 m2
            (F.col(feats[3]) < F.col(feat_col)) &  # cocina < área total
            (F.col(feats[0]) <= F.col(feats[1])) &  # piso <= total plantas
            (F.col(feats[3]) >= 0)  # cocina no negativa
        )
        .withColumn(
            f"{feats[2]}_label",
            F.when(F.col(feats[2]) == -1, "apartamento")
            .when(F.col(feats[2]) > 6, "más de 6 habitaciones")
            .when(F.col(feats[2]) == 1, "una habitación")
            .when(F.col(feats[2]) == 2, "dos habitaciones")
            .when(F.col(feats[2]) == 3, "tres habitaciones")
            .when(F.col(feats[2]) == 4, "cuatro habitaciones")
            .when(F.col(feats[2]) == 5, "cinco habitaciones")
            .when(F.col(feats[2]) == 6, "seis habitaciones")
            .otherwise("desconocido")
            )
        .withColumn(
            f"{feats[0]}_group",
            F.when(F.col(feats[0]) == 0, "planta baja")
            .when(F.col(feats[0]) > 30, "rascacielos")
            .when(F.col(feats[0]).between(20, 30), "de 20 a 30 plantas")
            .when(F.col(feats[0]).between(15, 19), "de 15 a 20 plantas")
            .when(F.col(feats[0]).between(10, 14), "de 10 a 15 plantas")
            .when(F.col(feats[0]).between(8, 9), "de 8 a 10 plantas")
            .when(F.col(feats[0]).between(6, 7), "de 6 a 8 plantas")
            .otherwise(
                F.concat(F.lit("planta "), F.col(feats[0]))
            )
        )
        .withColumn(
            f"{feats[1]}_group",
            F.when(F.col(feats[1]) == 0, "planta baja")
            .when(F.col(feats[1]) > 30, "rascacielos")
            .when(F.col(feats[1]).between(20, 30), "de 20 a 30 plantas")
            .when(F.col(feats[1]).between(15, 19), "de 15 a 20 plantas")
            .when(F.col(feats[1]).between(10, 14), "de 10 a 15 plantas")
            .when(F.col(feats[1]).between(8, 9), "de 8 a 10 plantas")
            .when(F.col(feats[1]).between(6, 7), "de 6 a 8 plantas")
            .otherwise(
                F.concat(F.lit("planta "), F.col(feats[1]))
            )
        )
        .withColumn("open_kitchen", F.when(F.col(feats[3]) == 0, "yes").otherwise("no"))
        .withColumn(feats[4], F.when(F.col(feats[4]) == 0, "mercado_secundario").otherwise("nueva_construccion"))
        .withColumn(
            feats[5],
            F.when(F.col(feats[5]) == 0, "desconocido")
            .when(F.col(feats[5]) == 1, "otro")
            .when(F.col(feats[5]) == 2, "panel")
            .when(F.col(feats[5]) == 3, "monolítico")
            .when(F.col(feats[5]) == 4, "ladrillo")
            .when(F.col(feats[5]) == 5, "bloques")
            .when(F.col(feats[5]) == 6, "madera")
            .otherwise("Desconocido")
        )
    ).drop(feats[0], feats[1], feats[2])

    logger.info("Filtrado de muestra y generación de variables")

    # filtro precio del inmueble mayor que cero
    if target_col is not None:
        logger.info(
            "Generación target: "
            f"Cálculo del precio por m2 y revisión de outliers. Uso de logartimo para su modelización"
        )
        df = (
            df.filter((F.col(target_col) > 0))
            .withColumn("price_m2", F.col(target_col) / F.col(feat_col))
            .filter(F.col("price_m2").between(*price_m2_range))  # revisar
            .withColumn("target", F.log(F.col("price_m2")))
        )
    logger.info("Realizado preprocesado de la muestra de datos")
    return df


def select_cols(
        feat_col: Literal["area"] = "area",
        model_features: Optional[List[str]] = None,
        target_col: Optional[Literal["price"]] = "price",
) -> List[str]:
    """
    Generación de lista de variables a incluir en etapa procesado de datos
    """
    feats_ = features if model_features is None else model_features
    feats = feats_ + [feat_col] if target_col is None else feats_ + [feat_col] + [target_col]
    return feats
