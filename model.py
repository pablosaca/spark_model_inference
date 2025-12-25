from model.session import spark_session
from model.data import load_data
from model.preprocessing import preprocessing_data
from model.train import train_test_split, model_train, model_evaluation
from model.save_model import save_artifact, save_metrics
from utils.utils import get_logger

logger = get_logger()


def main():

    logger.info("Inicio job de entrenamiento del modelo de spark")
    # creación de la sesión de spark
    spark = spark_session()

    # carga de datos y preprocesado
    df = load_data(spark)
    df = preprocessing_data(df)

    # Entrenamiento del modelo y evaluación en muestra de entrenamiento
    train_df, val_df = train_test_split(df)

    # Entrenamiento del modelo y evaluación en muestra de entrenamiento
    train_df.cache()
    model = model_train(train_df)
    train_metric_value = model_evaluation(model, train_df, sample_name="train")
    train_df.unpersist()

    # Evaluación de la muestra de test
    val_metric_value = model_evaluation(model, val_df, sample_name="test")

    # Resumen métricas
    metric_value = {**train_metric_value, **val_metric_value}
    del train_metric_value, val_metric_value
    save_metrics(metric_value)

    # Guardado del modelo
    save_artifact(spark, model=model)

    spark.stop()  # cierre de la sesión de spark
    logger.info("Se apaga la sesión de spark")


if __name__ == "__main__":
    main()
