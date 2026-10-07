import json
from pathlib import Path

from app.schema import FEATURE_ORDER


def load_reference(path: Path) -> dict:
    with path.open() as file:
        reference = json.load(file)

    if list(reference) != FEATURE_ORDER:
        raise ValueError("Reference feature order does not match model feature order.")

    return reference


def check_input_ranges(features: dict, reference: dict) -> dict:
    warnings = []

    for feature in FEATURE_ORDER:
        value = float(features[feature])
        q10 = float(reference[feature]["q10"])
        q90 = float(reference[feature]["q90"])

        if value < q10 or value > q90:
            warnings.append({
                "feature": feature,
                "value": value,
                "q10": q10,
                "q90": q90,
                "status": "outside_training_q10_q90",
            })

    return {
        "reference_type": "training_q10_q90_range_check",
        "warning_count": len(warnings),
        "warnings": warnings,
    }
