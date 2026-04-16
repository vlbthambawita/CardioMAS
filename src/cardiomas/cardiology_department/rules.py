from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ECGQualityRule(BaseModel):
    rule_id: str
    title: str
    description: str
    severity: str
    recommendation: str


class SplitRecommendation(BaseModel):
    strategy: str
    rationale: str
    candidate_label_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CardiologyReview(BaseModel):
    dataset_name: str
    summary_signals: dict[str, bool | list[str]]
    quality_rules: list[ECGQualityRule] = Field(default_factory=list)
    split_recommendation: SplitRecommendation
    exclusion_rules: list[str] = Field(default_factory=list)


DEFAULT_ECG_RULES = [
    ECGQualityRule(
        rule_id="ecg_qc_01",
        title="Signal completeness",
        description="Reject records with missing leads, unreadable waveforms, or empty samples.",
        severity="high",
        recommendation="Exclude incomplete signals before model training.",
    ),
    ECGQualityRule(
        rule_id="ecg_qc_02",
        title="Sampling-rate consistency",
        description="Track mixed sampling rates and document any resampling step explicitly.",
        severity="medium",
        recommendation="Keep original sampling-rate metadata and avoid silent resampling.",
    ),
    ECGQualityRule(
        rule_id="ecg_qc_03",
        title="Patient-level leakage",
        description="Avoid placing records from the same patient in different splits.",
        severity="high",
        recommendation="Use patient-stratified splits when patient identifiers exist.",
    ),
    ECGQualityRule(
        rule_id="ecg_qc_04",
        title="Label traceability",
        description="Each diagnostic label should map back to a documented source column or rule.",
        severity="medium",
        recommendation="Keep label provenance in the generated reports.",
    ),
]


def build_cardiology_review(dataset_name: str, dataset_summary_path: str, output_root: str) -> tuple[CardiologyReview, dict[str, str]]:
    summary_data = json.loads(Path(dataset_summary_path).read_text(encoding="utf-8"))
    csv_schemas = summary_data.get("csv_schemas", {})
    flattened_headers = {header.lower() for headers in csv_schemas.values() for header in headers}

    patient_like_fields = sorted(header for header in flattened_headers if "patient" in header)
    label_like_fields = sorted(
        header for header in flattened_headers if any(token in header for token in ("label", "target", "diagnos"))
    )

    has_patient_id = bool(patient_like_fields)
    strategy = "patient_stratified" if has_patient_id else "record_stratified"
    warnings = [] if has_patient_id else ["Patient identifier not detected; review leakage risk manually."]

    review = CardiologyReview(
        dataset_name=dataset_name,
        summary_signals={
            "has_patient_identifier": has_patient_id,
            "has_label_columns": bool(label_like_fields),
            "candidate_label_fields": label_like_fields,
        },
        quality_rules=DEFAULT_ECG_RULES,
        split_recommendation=SplitRecommendation(
            strategy=strategy,
            rationale="Prefer patient-level isolation for ECG research whenever patient linkage is available.",
            candidate_label_fields=label_like_fields,
            warnings=warnings,
        ),
        exclusion_rules=[
            "Exclude records with missing core identifiers or obviously broken metadata rows.",
            "Flag low-quality ECGs and keep the exclusion rule in the final report.",
            "Document any clinically motivated exclusion before generating ML splits.",
        ],
    )

    base = Path(output_root) / "reports" / dataset_name
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "ecg_review.json"
    md_path = base / "ecg_review.md"

    json_path.write_text(json.dumps(review.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(review), encoding="utf-8")

    return review, {
        "ecg_review.json": str(json_path),
        "ecg_review.md": str(md_path),
    }


def _render_markdown(review: CardiologyReview) -> str:
    lines = [
        f"# Cardiology Review: {review.dataset_name}",
        "",
        f"- Split strategy: `{review.split_recommendation.strategy}`",
        f"- Candidate label fields: {', '.join(review.split_recommendation.candidate_label_fields) or 'none detected'}",
        "",
        "## ECG Quality Rules",
        "",
    ]
    for rule in review.quality_rules:
        lines.append(f"- `{rule.rule_id}` {rule.title}: {rule.recommendation}")
    lines.extend(["", "## Exclusion Rules", ""])
    for rule in review.exclusion_rules:
        lines.append(f"- {rule}")
    if review.split_recommendation.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in review.split_recommendation.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"
