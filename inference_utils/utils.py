from pydantic import BaseModel
from typing import Optional, Dict, Union, Tuple


class PredictionResponse(BaseModel):
    model_prediccion: Optional[float]
    price: Optional[float]
    error: bool
    error_msg: Optional[str]


input_cols = [
    "area",
    "kitchen_area",
    "building_type",
    "object_type",
    "rooms",
    "level",
    "levels",
]


def validate_input(d: Dict[str, Union[int, float]]) -> Tuple[bool, str]:

    if not (d["area"] > 10):
        return False, "El área total de la casa debe ser mayor a 10"

    if not (0 <= d["kitchen_area"] < d.get("area", 0)):
        return False, "El área de la cocina debe ser positiva y menor al área total"

    if not (d["level"] <= d["levels"]):
        return False, "El piso no puede ser mayor que el número de plantas del inmueble"
    return True, ""


def rooms_label(rooms: int) -> str:
    if rooms == -1:
        return "apartamento"
    if rooms > 6:
        return "más de 6 habitaciones"
    mapping = {
        1: "una habitación",
        2: "dos habitaciones",
        3: "tres habitaciones",
        4: "cuatro habitaciones",
        5: "cinco habitaciones",
        6: "seis habitaciones"
    }
    return mapping.get(rooms, "desconocido")


def level_group(level: int) -> str:
    if level == 0:
        return "planta baja"
    if level > 30:
        return "rascacielos"
    if 20 <= level <= 30:
        return "de 20 a 30 plantas"
    if 15 <= level <= 19:
        return "de 15 a 20 plantas"
    if 10 <= level <= 14:
        return "de 10 a 15 plantas"
    if 8 <= level <= 9:
        return "de 8 a 10 plantas"
    if 6 <= level <= 7:
        return "de 6 a 8 plantas"
    return f"planta {level}"


def open_kitchen(kitchen_area: float) -> str:
    return "yes" if kitchen_area == 0 else "no"


def object_type_label(v: int) -> str:
    return "mercado_secundario" if v == 0 else "nueva_construccion"


def rooms_label(rooms: int) -> str:
    if rooms == -1:
        return "apartamento"
    if rooms > 6:
        return "más de 6 habitaciones"
    mapping = {
        1: "una habitación",
        2: "dos habitaciones",
        3: "tres habitaciones",
        4: "cuatro habitaciones",
        5: "cinco habitaciones",
        6: "seis habitaciones"
    }
    return mapping.get(rooms, "desconocido")


def building_type_label(v: int) -> str:
    mapping = {
        0: "desconocido",
        1: "otro",
        2: "panel",
        3: "monolítico",
        4: "ladrillo",
        5: "bloques",
        6: "madera"
    }
    return mapping.get(v, "desconocido")
