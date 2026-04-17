from __future__ import annotations

from pathlib import Path

from cardiomas.agentic.runtime import AgenticRuntime
from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.coding.tool_builder import build_generated_tool_package as real_build_generated_tool_package
from cardiomas.schemas.config import AutonomyConfig, KnowledgeSource, RuntimeConfig
from cardiomas.schemas.tools import ToolSpec


def test_runtime_query_can_generate_dataset_statistics_tool(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text(
        "record_id,patient_id,label,age\nrec_1,pat_1,NORM,43\nrec_2,pat_2,AFIB,54\nrec_3,pat_3,NORM,50\n",
        encoding="utf-8",
    )

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[KnowledgeSource(kind="dataset_dir", path=str(dataset_dir), label="dataset")],
        autonomy=AutonomyConfig(enable_code_agents=True, allow_tool_codegen=True),
    )

    result = AgenticRuntime(config).query("Give me summary statistics for this dataset.", force_rebuild=True)

    assert any(call.tool_name == "dataset_statistics" for call in result.tool_calls)
    assert result.repair_traces
    assert any(trace.tool_name == "dataset_statistics" and trace.retry_succeeded for trace in result.repair_traces)
    assert "Dataset statistics:" in result.answer
    assert "rows=3" in result.answer
    assert (Path(config.output_dir) / "autonomy_workspace" / "tools" / "dataset_statistics" / "tool.py").exists()


def test_runtime_query_can_generate_shell_script(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,label\nrec_1,NORM\n", encoding="utf-8")

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[KnowledgeSource(kind="dataset_dir", path=str(dataset_dir), label="dataset")],
        autonomy=AutonomyConfig(enable_code_agents=True, allow_script_codegen=True),
    )

    result = AgenticRuntime(config).query("Write a shell script to inspect this dataset.", force_rebuild=True)

    assert any(call.tool_name == "generate_shell_script" for call in result.tool_calls)
    assert any(trace.tool_name == "generate_shell_script" and trace.retry_succeeded for trace in result.repair_traces)
    script_path = Path(result.evidence[0].uri)
    assert script_path.exists()
    assert "Generated shell script:" in result.answer


def test_autonomous_tool_manager_repairs_after_invalid_generation(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,label\nrec_1,NORM\n", encoding="utf-8")

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[KnowledgeSource(kind="dataset_dir", path=str(dataset_dir), label="dataset")],
        autonomy=AutonomyConfig(enable_code_agents=True, allow_tool_codegen=True, max_repair_attempts=1),
    )

    calls = {"count": 0}

    def flaky_builder(tool_name: str, payload: dict, config: RuntimeConfig, last_error: str = ""):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                ToolSpec(name=tool_name, description="bad tool", category="autonomy", generated=True),
                "import subprocess\n\ndef run(payload):\n    return {'ok': True, 'summary': 'bad', 'data': {}}\n",
                "# bad tool",
            )
        return real_build_generated_tool_package(tool_name, payload, config, last_error=last_error)

    monkeypatch.setattr("cardiomas.autonomy.recovery.build_generated_tool_package", flaky_builder)

    manager = AutonomousToolManager(config)
    result = manager.dataset_statistics(dataset_path=str(dataset_dir))
    traces = manager.consume_traces()

    assert result.ok is True
    assert len(traces) == 2
    assert traces[0].ok is False
    assert "Banned module imported" in traces[0].error
    assert traces[1].retry_succeeded is True
