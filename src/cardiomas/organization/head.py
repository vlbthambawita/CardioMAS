from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from cardiomas.cardiology_department.pipeline import CardiologyDepartmentHead
from cardiomas.coding_department.pipeline import CodingDepartmentHead
from cardiomas.knowledge_department.pipeline import KnowledgeDepartmentHead
from cardiomas.organization.base import DepartmentHead
from cardiomas.shared.messages import ApprovalRecord, ArtifactRef, ArtifactStatus, DepartmentReport, TaskMessage
from cardiomas.testing_department.pipeline import TestingDepartmentHead


class OrganizationRunResult(BaseModel):
    dataset_name: str
    goal: str
    output_dir: str
    status: str
    department_reports: list[DepartmentReport] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    communication_log: list[TaskMessage] = Field(default_factory=list)
    final_artifacts: list[ArtifactRef] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class OrganizationHead:
    """Top-level coordinator that owns all inter-department communication."""

    def __init__(self, departments: dict[str, DepartmentHead], name: str = "organization_head") -> None:
        self.name = name
        self.departments = departments

    def run(
        self,
        goal: str,
        dataset_name: str,
        dataset_dir: str,
        knowledge_urls: list[str] | None = None,
        output_dir: str = "organization_output",
        approve: bool = False,
    ) -> OrganizationRunResult:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        reports: list[DepartmentReport] = []
        approvals: list[ApprovalRecord] = []
        messages: list[TaskMessage] = []
        errors: list[str] = []
        final_artifacts: list[ArtifactRef] = []

        knowledge_urls = knowledge_urls or []
        knowledge_artifacts: list[ArtifactRef] = []
        coding_artifacts: list[ArtifactRef] = []

        if knowledge_urls:
            knowledge_report, knowledge_message = self._dispatch(
                department="knowledge",
                purpose="Collect reusable dataset knowledge",
                instructions="Fetch the provided dataset pages, normalize the content, and write reusable knowledge artifacts.",
                inputs={
                    "dataset_name": dataset_name,
                    "dataset_dir": dataset_dir,
                    "knowledge_urls": knowledge_urls,
                    "output_root": str(root),
                },
            )
            reports.append(knowledge_report)
            messages.append(knowledge_message)
            knowledge_artifacts = knowledge_report.artifacts
            final_artifacts.extend(knowledge_artifacts)
        else:
            reports.append(
                DepartmentReport(
                    department="knowledge",
                    summary="No knowledge URLs were provided; skipped web knowledge collection.",
                    notes=["Pass one or more --knowledge-url values to generate reusable knowledge files."],
                )
            )

        coding_report, coding_message = self._dispatch(
            department="coding",
            purpose="Build reusable dataset-analysis outputs",
            instructions="Analyze the local dataset directory and write machine-readable plus human-readable outputs.",
            inputs={
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "output_root": str(root),
                "knowledge_artifacts": [artifact.model_dump(mode="json") for artifact in knowledge_artifacts],
            },
        )
        reports.append(coding_report)
        messages.append(coding_message)
        coding_artifacts = coding_report.artifacts
        final_artifacts.extend(coding_artifacts)

        summary_json = next((artifact.path for artifact in coding_artifacts if artifact.name == "dataset_inventory_json"), "")

        cardiology_report, cardiology_message = self._dispatch(
            department="cardiology",
            purpose="Review ECG-specific quality and split assumptions",
            instructions="Review the dataset summary and emit ECG quality rules plus split recommendations.",
            inputs={
                "dataset_name": dataset_name,
                "dataset_summary_path": summary_json,
                "output_root": str(root),
            },
        )
        reports.append(cardiology_report)
        messages.append(cardiology_message)
        final_artifacts.extend(cardiology_report.artifacts)

        testing_report, testing_message = self._dispatch(
            department="testing",
            purpose="Validate coding outputs and reproducibility",
            instructions="Validate output existence, schema shape, reproducibility, and invalid-path handling. Produce structured reports.",
            inputs={
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "output_root": str(root),
                "coding_artifacts": [artifact.model_dump(mode="json") for artifact in coding_artifacts],
            },
        )
        reports.append(testing_report)
        messages.append(testing_message)
        final_artifacts.extend(testing_report.artifacts)

        for report in reports:
            for artifact in report.artifacts:
                artifact.status = ArtifactStatus.approved if approve else ArtifactStatus.pending
                approvals.append(
                    ApprovalRecord(
                        artifact_name=artifact.name,
                        artifact_path=artifact.path,
                        status=artifact.status,
                        notes="Approved by Organization Head." if approve else "Awaiting Organization Head approval.",
                    )
                )

        for report in reports:
            if report.summary.lower().startswith("failed"):
                errors.append(f"{report.department}: {report.summary}")

        status = "approved" if approve and not errors else "awaiting_approval"
        if errors:
            status = "failed"

        return OrganizationRunResult(
            dataset_name=dataset_name,
            goal=goal,
            output_dir=str(root),
            status=status,
            department_reports=reports,
            approvals=approvals,
            communication_log=messages,
            final_artifacts=final_artifacts,
            errors=errors,
        )

    def _dispatch(
        self,
        department: str,
        purpose: str,
        instructions: str,
        inputs: dict,
    ) -> tuple[DepartmentReport, TaskMessage]:
        if department not in self.departments:
            raise KeyError(f"Unknown department: {department}")

        head = self.departments[department]
        message = TaskMessage(
            sender=self.name,
            recipient=head.name,
            department=department,
            purpose=purpose,
            instructions=instructions,
            inputs=inputs,
            approval_required=True,
        )
        report = head.handle_task(message)
        return report, message


def build_default_organization() -> OrganizationHead:
    return OrganizationHead(
        departments={
            "knowledge": KnowledgeDepartmentHead(),
            "coding": CodingDepartmentHead(),
            "cardiology": CardiologyDepartmentHead(),
            "testing": TestingDepartmentHead(),
        }
    )
