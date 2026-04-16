from __future__ import annotations

from cardiomas.coding_department.contracts import default_tool_contracts
from cardiomas.coding_department.tools import summarize_dataset_directory, write_dataset_summary
from cardiomas.organization.base import DepartmentHead, WorkerAgent
from cardiomas.shared.messages import ArtifactRef, ArtifactStatus, DepartmentReport, TaskMessage


class CodingWorker(WorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="coding_worker", department_name="coding")

    def execute(self, message: TaskMessage) -> DepartmentReport:
        dataset_name = message.inputs["dataset_name"]
        dataset_dir = message.inputs["dataset_dir"]
        output_root = message.inputs["output_root"]

        summary = summarize_dataset_directory(dataset_name, dataset_dir)
        output_paths = write_dataset_summary(summary, output_root)

        artifacts = [
            ArtifactRef(
                name="dataset_inventory_json",
                path=output_paths["dataset_inventory.json"],
                artifact_type="json",
                summary="Machine-readable dataset inventory",
                status=ArtifactStatus.complete,
            ),
            ArtifactRef(
                name="file_extensions_csv",
                path=output_paths["file_extensions.csv"],
                artifact_type="csv",
                summary="Extension counts for the dataset",
                status=ArtifactStatus.complete,
            ),
            ArtifactRef(
                name="dataset_inventory_markdown",
                path=output_paths["dataset_inventory.md"],
                artifact_type="markdown",
                summary="Human-readable dataset inventory report",
                status=ArtifactStatus.complete,
            ),
        ]

        contracts = default_tool_contracts()
        return DepartmentReport(
            department="coding",
            summary=f"Generated {len(artifacts)} analysis artifact(s) using {len(contracts)} reusable tool contract(s).",
            artifacts=artifacts,
            notes=[contract.cli_command for contract in contracts],
        )


class CodingDepartmentHead(DepartmentHead):
    def __init__(self) -> None:
        super().__init__(
            name="coding_head",
            department_name="coding",
            workers=[CodingWorker()],
            input_contract="dataset_name, dataset_dir, output_root, knowledge_artifacts[]",
            output_contract="dataset_inventory.json, file_extensions.csv, dataset_inventory.md",
        )
