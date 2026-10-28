"""Database-free prediction sandbox endpoints."""

from copy import deepcopy

from fastapi import APIRouter, Depends, Query

from api.routers.pipelines import get_predictor
from api.schemas import (
    BuildFeaturesIn,
    OptimizationSimulationIn,
    OptimizationSimulationOut,
)
from src.inference.predictor import CICDPredictor


router = APIRouter(prefix="/predict", tags=["prediction"])

OPTIMIZATION_DISCLAIMER = (
    "Model-based simulation only. It does not guarantee the actual build outcome or failure rate."
)


def _scenario_output(predictor: CICDPredictor, build_features: dict, include_explanation: bool) -> dict:
    """Score one transient scenario through the existing frozen predictor path."""
    analysis = predictor.full_analysis(build_features, include_explanation=include_explanation)
    if getattr(predictor, "input_schema", "travis") == "github_actions":
        derived = {"total_churn": float(predictor._tier_e_frame(build_features).loc[0, "total_churn"])}
    else:
        derived = {"total_files_changed": float(predictor._prepared_features(build_features).loc[0, "total_files_changed"])}
    return {
        "failure_probability": analysis["prediction"]["failure_probability"],
        "predicted_status": analysis["prediction"]["predicted_status"],
        "derived_features": derived,
        "explanation": analysis.get("explanation") if include_explanation else None,
    }


@router.post("/whatif")
def what_if_prediction(
    payload: BuildFeaturesIn,
    explain: bool = Query(False),
    predictor: CICDPredictor = Depends(get_predictor),
) -> dict:
    """Score hypothetical features and flag unusual pre-build input profiles without persistence."""
    prediction = predictor.predict_failure(payload.build_features)
    anomaly = predictor.detect_anomaly(payload.build_features)
    result = {
        "failure_probability": prediction["failure_probability"],
        "is_anomaly": anomaly["is_anomaly"],
    }
    if explain:
        result["explanation"] = predictor.full_analysis(payload.build_features, include_explanation=True)["explanation"]
    return result


@router.post("/whatif/optimization", response_model=OptimizationSimulationOut)
def optimization_what_if(
    payload: OptimizationSimulationIn,
    predictor: CICDPredictor = Depends(get_predictor),
) -> dict:
    """Compare one bounded hypothetical input with its unpersisted baseline."""
    baseline_features = deepcopy(payload.build_features)
    simulated_features = deepcopy(baseline_features)
    field = payload.intervention.field
    original_value = simulated_features.get(field, 0.0)
    simulated_features[field] = payload.intervention.value

    baseline = _scenario_output(predictor, baseline_features, payload.include_explanation)
    simulation = _scenario_output(predictor, simulated_features, False)
    risk_delta = simulation["failure_probability"] - baseline["failure_probability"]
    return {
        "simulation_type": "model_based_hypothetical",
        "baseline": baseline,
        "simulation": simulation,
        "risk_delta": risk_delta,
        "percentage_point_change": risk_delta * 100,
        "intervention": {
            "field": field,
            "original_value": original_value,
            "simulated_value": payload.intervention.value,
            "reason": payload.intervention.reason,
        },
        "disclaimer": OPTIMIZATION_DISCLAIMER,
    }
