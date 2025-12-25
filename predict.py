from pathlib import Path

from fastapi import FastAPI
import onnxruntime as rt
import numpy as np

from inference_utils.utils import (
    PredictionResponse,
    building_type_label,
    object_type_label,
    rooms_label,
    level_group,
    open_kitchen,
    validate_input,
    input_cols,
)

from utils.utils import get_logger

logger = get_logger()


app = FastAPI()

# se carga el modelo globalmente para que sea rápido ONNX Runtime es ultra eficiente

base_path = Path(__file__).resolve().parent
model_path = str(base_path / "artifact" / "modelo_precios.onnx")

sess = rt.InferenceSession(model_path)
logger.info(f"El modelo ha sido cargado: {model_path}")


@app.post("/spark_model", response_model=PredictionResponse)
async def predict(data: dict):
    """
    Input ejemplo
    """

    missing_keys = [k for k in input_cols if k not in data.keys()]

    logger.info(f"Chequeo las claves del json son las necesarias para hacer inferencia")
    if missing_keys:
        logger.info(f"Error al hacer inferencia : Faltan las siguientes keys en el input: {', '.join(missing_keys)}")
        return PredictionResponse(
            model_prediccion=None,
            price=None,
            error=True,
            error_msg=f"Faltan las siguientes keys en el input: {', '.join(missing_keys)}"
        )

    logger.info(f"Chequeo inputs valores iniciales sobre las variables para hacer inferencia")
    is_valid, msg = validate_input(data)
    if not is_valid:
        logger.info(msg)
        return PredictionResponse(
            model_prediccion=None,
            price=None,
            error=True,
            error_msg=msg
        )

    logger.info(f"Construcción de las variables necesarias para hacer la inferencia")
    features = {
        "kitchen_area": float(data["kitchen_area"]),
        "building_type": building_type_label(data["building_type"]),
        "object_type": object_type_label(data["object_type"]),
        "rooms_label": rooms_label(data["rooms"]),
        "level_group": level_group(data["level"]),
        "levels_group": level_group(data["levels"]),
        "open_kitchen": open_kitchen(data["kitchen_area"])
    }

    inputs = {}
    logger.info(f"Revisión de tipado de las variables para realizar inferencia")
    for key in features.keys():
        value = features[key]
        if isinstance(value, str):
            inputs[key] = np.array([[value]], dtype=object)
        else:
            inputs[key] = np.array([[value]], dtype=np.float32)

    # El resultado es una lista de array
    res = sess.run(None, inputs)

    # La salida de un modelo de regresión
    prediction = np.exp(res[0][0][0])
    logger.info(
        f"Inferencia realizada. Cálculo del precio/m2 y el precio total de la vivienda según el área {data['area']}"
    )
    return PredictionResponse(
        model_prediccion=prediction,
        price=prediction * data["area"],
        error=False,
        error_msg=None
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
