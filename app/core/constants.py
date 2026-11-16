"""Product-level constants kept independent from the ML runtime."""

from __future__ import annotations

CLASS_NAMES: dict[int, str] = {
    0: "obstrucao_visivel",
    1: "agua_acumulada",
    2: "animal_visivel",
}
CLASS_IDS: dict[str, int] = {name: class_id for class_id, name in CLASS_NAMES.items()}

PUBLIC_CLASS_LABELS: dict[str, str] = {
    "obstrucao_visivel": "Obstrução visível",
    "agua_acumulada": "Água acumulada",
    "animal_visivel": "Animal visível",
}

DISCLAIMER = (
    "Triagem assistida por IA. A detecção não confirma risco sanitário ou "
    "hidráulico e não substitui inspeção humana qualificada."
)
MODEL_UNAVAILABLE_DETAIL = (
    "Modelo treinado indisponível. Treine o modelo ou configure "
    "DRENAVISION_MODEL_PATH para um checkpoint válido."
)
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
}
