from __future__ import annotations

from cardiomas.cardiology_department.rules import build_cardiology_review
from cardiomas.organization.base import DepartmentHead, WorkerAgent
from cardiomas.shared.messages import ArtifactRef, ArtifactStatus, DepartmentReport, TaskMessage


class CardiologyWorker(WorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="cardiology_worker", department_name="cardiology")

    def execute(self, message: TaskMessage) -> DepartmentReport:
        dataset_name = message.inputs["dataset_name"]
        dataset_summary_path = message.inputs["dataset_summary_path"]
        output_root = message.inputs["output_root"]

        _, output_paths = build_cardiology_review(dataset_name, dataset_summary_path, output_root)

        artifacts = [
            ArtifactRef(
                name="ecg_review_json",
                path=output_paths["ecg_review.json"],
                artifact_type="json",
                summary="Machine-readable ECG review",
                status=ArtifactStatus.complete,
            ),
            ArtifactRef(
                name="ecg_review_markdown",
                path=output_paths["ecg_review.md"],
                artifact_type="markdown",
                summary="Human-readable ECG review",
                status=ArtifactStatus.complete,
            ),
        ]

        return DepartmentReport(
            department="cardiology",
            summary="Produced ECG quality and split recommendations.",
            artifacts=artifacts,
        )


class CardiologyDepartmentHead(DepartmentHead):
    def __init__(self) -> None:
        super().__init__(
            name="cardiology_head",
            department_name="cardiology",
            workers=[CardiologyWorker()],
            input_contract="dataset_name, dataset_summary_path, output_root",
            output_contract="ecg_review.json, ecg_review.md",
        )
