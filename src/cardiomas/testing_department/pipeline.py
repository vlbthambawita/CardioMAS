from __future__ import annotations

import json
from pathlib import Path

from cardiomas.coding_department.tools import summarize_dataset_directory
from cardiomas.organization.base import DepartmentHead, WorkerAgent
from cardiomas.shared.messages import ArtifactRef, ArtifactStatus, DepartmentReport, TaskMessage
from cardiomas.testing_department.reports import ToolTestReport


def validate_coding_outputs(dataset_name: str, dataset_dir: str, coding_artifacts: list[dict], output_root: str) -> tuple[ToolTestReport, dict[str, str]]:
    artifact_map = {artifact["name"]: artifact["path"] for artifact in coding_artifacts}
    checks = {
        "json_exists": Path(artifact_map.get("dataset_inventory_json", "")).exists(),
        "csv_exists": Path(artifact_map.get("file_extensions_csv", "")).exists(),
        "markdown_exists": Path(artifact_map.get("dataset_inventory_markdown", "")).exists(),
    }
    details: list[str] = []

    json_path = Path(artifact_map.get("dataset_inventory_json", ""))
    if json_path.exists():
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        required_keys = {"dataset_name", "dataset_dir", "total_files", "extension_counts", "csv_schemas"}
        checks["json_schema"] = required_keys.issubset(stored.keys())
        rerun = summarize_dataset_directory(dataset_name, dataset_dir)
        checks["reproducible"] = (
            stored.get("total_files") == rerun.total_files
            and stored.get("extension_counts") == rerun.extension_counts
            and stored.get("csv_schemas") == rerun.csv_schemas
        )
        details.append(f"Stored total_files={stored.get('total_files')}, rerun total_files={rerun.total_files}")
    else:
        checks["json_schema"] = False
        checks["reproducible"] = False

    try:
        summarize_dataset_directory(dataset_name, str(Path(dataset_dir) / "__missing__"))
    except FileNotFoundError:
        checks["invalid_path_handling"] = True
    else:
        checks["invalid_path_handling"] = False

    report = ToolTestReport(
        tool_name="dataset_inventory",
        passed=all(checks.values()),
        checks=checks,
        details=details,
    )

    base = Path(output_root) / "reports" / dataset_name
    base.mkdir(parents=True, exist_ok=True)
    json_report = base / "tool_validation.json"
    md_report = base / "tool_validation.md"
    json_report.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_report.write_text(_render_markdown(report), encoding="utf-8")

    return report, {
        "tool_validation.json": str(json_report),
        "tool_validation.md": str(md_report),
    }


def _render_markdown(report: ToolTestReport) -> str:
    lines = [f"# Tool Validation: {report.tool_name}", ""]
    for check_name, passed in report.checks.items():
        lines.append(f"- `{check_name}`: {'PASS' if passed else 'FAIL'}")
    if report.details:
        lines.extend(["", "## Details", ""])
        for detail in report.details:
            lines.append(f"- {detail}")
    return "\n".join(lines).rstrip() + "\n"


class TestingWorker(WorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="testing_worker", department_name="testing")

    def execute(self, message: TaskMessage) -> DepartmentReport:
        dataset_name = message.inputs["dataset_name"]
        dataset_dir = message.inputs["dataset_dir"]
        output_root = message.inputs["output_root"]
        coding_artifacts = list(message.inputs.get("coding_artifacts", []))

        report, output_paths = validate_coding_outputs(dataset_name, dataset_dir, coding_artifacts, output_root)
        artifacts = [
            ArtifactRef(
                name="tool_validation_json",
                path=output_paths["tool_validation.json"],
                artifact_type="json",
                summary="Machine-readable validation report",
                status=ArtifactStatus.complete,
            ),
            ArtifactRef(
                name="tool_validation_markdown",
                path=output_paths["tool_validation.md"],
                artifact_type="markdown",
                summary="Human-readable validation report",
                status=ArtifactStatus.complete,
            ),
        ]
        return DepartmentReport(
            department="testing",
            summary="Validation passed." if report.passed else "Validation found issues.",
            artifacts=artifacts,
            notes=report.details,
        )


class TestingDepartmentHead(DepartmentHead):
    def __init__(self) -> None:
        super().__init__(
            name="testing_head",
            department_name="testing",
            workers=[TestingWorker()],
            input_contract="dataset_name, dataset_dir, output_root, coding_artifacts[]",
            output_contract="tool_validation.json, tool_validation.md",
        )
