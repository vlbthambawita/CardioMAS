"""
V2 feature tests:
- Per-agent LLM config
- Session recorder
- ParsedRequirement schema
- Orchestrator routing logic
- NL requirement agent (mock LLM)
- Coder agent script generation (mock LLM)
- Hub-and-spoke workflow routing
- Context compression helper
- cardiomas resume CLI
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Per-agent LLM config ──────────────────────────────────────────────────

def test_get_agent_llm_default():
    import cardiomas.config as cfg
    model = cfg.get_agent_llm("discovery")
    assert model == cfg.OLLAMA_MODEL


def test_get_agent_llm_override():
    import cardiomas.config as cfg
    original = cfg._AGENT_LLM_OVERRIDES.copy()
    cfg.set_agent_llm("coder", "deepseek-coder:6.7b")
    assert cfg.get_agent_llm("coder") == "deepseek-coder:6.7b"
    # Restore
    cfg._AGENT_LLM_OVERRIDES.clear()
    cfg._AGENT_LLM_OVERRIDES.update(original)


def test_get_llm_for_agent_runtime_map():
    from cardiomas.llm_factory import get_llm_for_agent
    import cardiomas.config as cfg

    # With a runtime map that has the agent
    with patch("cardiomas.llm_factory.get_local_llm") as mock_local:
        mock_llm = MagicMock()
        mock_local.return_value = mock_llm
        llm = get_llm_for_agent("analysis", agent_llm_map={"analysis": "gemma3:4b"})
        mock_local.assert_called_once()
        call_kwargs = mock_local.call_args
        assert call_kwargs.kwargs.get("model") == "gemma3:4b" or call_kwargs.args[1] == "gemma3:4b"


def test_get_llm_for_agent_default_key_in_map():
    from cardiomas.llm_factory import get_llm_for_agent

    with patch("cardiomas.llm_factory.get_local_llm") as mock_local:
        mock_llm = MagicMock()
        mock_local.return_value = mock_llm
        get_llm_for_agent("discovery", agent_llm_map={"default": "gemma3:4b"})
        mock_local.assert_called_once()
        call_kwargs = mock_local.call_args
        model_used = call_kwargs.kwargs.get("model") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        assert model_used == "gemma3:4b"


# ── Session recorder ──────────────────────────────────────────────────────

def test_recorder_session_lifecycle():
    from cardiomas.recorder import SessionRecorder

    rec = SessionRecorder.reset()
    session_id = rec.start_session(
        dataset_name="ptb-xl",
        user_options={"seed": 42},
        raw_requirement="80/10/10 split",
    )
    assert len(session_id) == 36  # UUID

    rec.start_step("discovery", "identify_dataset", inputs={"source": "/data/ptb-xl"})
    rec.record_llm_call(
        agent="discovery",
        model="llama3.1:8b",
        system_prompt="You are ...",
        user_message="Identify this dataset",
        response="PTB-XL, 21799 records",
        duration_ms=1200,
    )
    rec.end_step(outputs={"name": "ptb-xl"}, reasoning="Registry match")

    log = rec.finish_session("ok")
    assert log is not None
    assert log.final_status == "ok"
    assert log.dataset_name == "ptb-xl"
    assert len(log.agent_steps) == 1
    step = log.agent_steps[0]
    assert step.agent == "discovery"
    assert len(step.llm_calls) == 1
    assert step.llm_calls[0].model == "llama3.1:8b"


def test_recorder_save(tmp_path):
    from cardiomas.recorder import SessionRecorder

    rec = SessionRecorder.reset()
    rec.start_session("ptb-xl", {})
    rec.start_step("splitter", "generate")
    rec.end_step(outputs={}, reasoning="Done")
    rec.add_orchestrator_reasoning("[start] → [discovery]: no cache hit")
    rec.finish_session("ok")

    files = rec.save(tmp_path)
    assert (tmp_path / "session_log" / "session.json").exists()
    assert (tmp_path / "session_log" / "conversation.md").exists()
    assert (tmp_path / "session_log" / "reasoning_trace.md").exists()

    session_data = json.loads((tmp_path / "session_log" / "session.json").read_text())
    assert session_data["dataset_name"] == "ptb-xl"
    assert session_data["final_status"] == "ok"


def test_recorder_markdown_content(tmp_path):
    from cardiomas.recorder import SessionRecorder

    rec = SessionRecorder.reset()
    rec.start_session("test-ds", {}, raw_requirement="80/10/10 split")
    rec.start_step("analysis", "scan")
    rec.record_llm_call("analysis", "gemma3:4b", "sys", "user msg", "response text")
    rec.end_step(reasoning="Found 5000 records")
    rec.add_orchestrator_reasoning("[start] → [discovery]: initial")
    rec.finish_session("ok")
    rec.save(tmp_path)

    conv = (tmp_path / "session_log" / "conversation.md").read_text()
    assert "User Requirement" in conv
    assert "80/10/10 split" in conv
    assert "gemma3:4b" in conv
    assert "response text" in conv

    trace = (tmp_path / "session_log" / "reasoning_trace.md").read_text()
    assert "[start] → [discovery]" in trace


# ── ParsedRequirement schema ──────────────────────────────────────────────

def test_parsed_requirement_defaults():
    from cardiomas.schemas.requirement import ParsedRequirement

    pr = ParsedRequirement(raw_input="80/10/10 split")
    assert pr.split_ratios == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert pr.patient_level is True
    assert pr.stratify_by is None
    assert pr.seed is None


def test_parsed_requirement_full():
    from cardiomas.schemas.requirement import ParsedRequirement

    pr = ParsedRequirement(
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        stratify_by="rhythm",
        exclusion_filters=[{"field": "age", "op": "gt", "value": 17}],
        patient_level=True,
        seed=99,
        raw_input="80/10/10, stratify by rhythm, no kids",
        llm_reasoning="Parsed 80/10/10 ratios.",
    )
    assert pr.split_ratios["train"] == 0.8
    assert pr.stratify_by == "rhythm"
    assert pr.seed == 99


# ── Orchestrator routing logic ────────────────────────────────────────────

def test_orchestrator_route_nl_requirement_to_discovery():
    from cardiomas.agents.orchestrator import _route
    from cardiomas.schemas.state import GraphState, UserOptions

    state = GraphState(
        dataset_source="/data/ptb-xl",
        user_options=UserOptions(dataset_source="/data/ptb-xl"),
    )
    next_a, reason = _route(state, "nl_requirement")
    assert next_a == "discovery"


def test_orchestrator_route_discovery_local_skips_paper():
    from cardiomas.agents.orchestrator import _route
    from cardiomas.schemas.state import GraphState, UserOptions

    state = GraphState(
        dataset_source="/data/ptb-xl",
        user_options=UserOptions(
            dataset_source="/data/ptb-xl",
            local_path="/data/ptb-xl",
        ),
    )
    next_a, reason = _route(state, "discovery")
    assert next_a == "analysis"
    assert "paper" not in next_a


def test_orchestrator_route_discovery_url_goes_to_paper():
    from cardiomas.agents.orchestrator import _route
    from cardiomas.schemas.state import GraphState, UserOptions

    state = GraphState(
        dataset_source="https://physionet.org/content/ptb-xl/1.0.3/",
        user_options=UserOptions(dataset_source="https://physionet.org/content/ptb-xl/1.0.3/"),
    )
    next_a, reason = _route(state, "discovery")
    assert next_a == "paper"


def test_orchestrator_route_security_fail():
    from cardiomas.agents.orchestrator import _route
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.schemas.audit import SecurityAudit

    state = GraphState(
        dataset_source="/data/test",
        user_options=UserOptions(dataset_source="/data/test"),
        security_audit=SecurityAudit(passed=False, blocking_issues=["PII detected"]),
    )
    next_a, reason = _route(state, "security")
    assert next_a == "end_with_error"


def test_orchestrator_route_coder_no_push():
    from cardiomas.agents.orchestrator import _route
    from cardiomas.schemas.state import GraphState, UserOptions

    state = GraphState(
        dataset_source="/data/test",
        user_options=UserOptions(dataset_source="/data/test", push_to_hf=False),
    )
    next_a, _ = _route(state, "coder")
    assert next_a == "end_saved"


def test_orchestrator_route_coder_with_push():
    from cardiomas.agents.orchestrator import _route
    from cardiomas.schemas.state import GraphState, UserOptions

    state = GraphState(
        dataset_source="/data/test",
        user_options=UserOptions(dataset_source="/data/test", push_to_hf=True),
    )
    next_a, _ = _route(state, "coder")
    assert next_a == "publisher"


def test_orchestrator_retry_on_error():
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.recorder import SessionRecorder
    from cardiomas.agents.orchestrator import orchestrator_agent

    SessionRecorder.reset().start_session("test", {})

    state = GraphState(
        dataset_source="/data/test",
        user_options=UserOptions(dataset_source="/data/test"),
        last_completed_agent="discovery",
        session_id="test-session",
        errors=["Connection timeout"],
    )
    result = orchestrator_agent(state)
    # First error → retry
    assert result.next_agent == "discovery"
    assert result.retry_counts.get("discovery", 0) == 1


def test_orchestrator_abort_after_retry():
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.recorder import SessionRecorder
    from cardiomas.agents.orchestrator import orchestrator_agent

    SessionRecorder.reset().start_session("test", {})

    state = GraphState(
        dataset_source="/data/test",
        user_options=UserOptions(dataset_source="/data/test"),
        last_completed_agent="discovery",
        session_id="test-session",
        errors=["Connection timeout"],
        retry_counts={"discovery": 1},  # already retried once
    )
    result = orchestrator_agent(state)
    assert result.next_agent == "end_with_error"


# ── NL requirement agent (mock LLM) ──────────────────────────────────────

def test_nl_requirement_agent_no_requirement():
    from cardiomas.agents.nl_requirement import nl_requirement_agent
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.recorder import SessionRecorder

    SessionRecorder.reset().start_session("test", {})
    state = GraphState(
        dataset_source="/data/ptb-xl",
        user_options=UserOptions(dataset_source="/data/ptb-xl", requirement=None),
    )
    result = nl_requirement_agent(state)
    assert result.parsed_requirement is None


def test_nl_requirement_agent_parses_json():
    from cardiomas.agents.nl_requirement import nl_requirement_agent
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.recorder import SessionRecorder

    SessionRecorder.reset().start_session("test", {})
    state = GraphState(
        dataset_source="/data/ptb-xl",
        user_options=UserOptions(
            dataset_source="/data/ptb-xl",
            requirement="80/10/10 split stratified by rhythm",
        ),
    )

    json_resp = json.dumps({
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "stratify_by": "rhythm",
        "exclusion_filters": [],
        "patient_level": True,
        "seed": None,
        "notes": "",
        "raw_input": "80/10/10 split stratified by rhythm",
        "llm_reasoning": "Parsed 80/10/10 directly.",
    })

    # Patch run_agent at the source module — avoids all LLM connection issues
    with patch("cardiomas.agents.base.run_agent", return_value=json_resp):
        result = nl_requirement_agent(state)

    assert result.parsed_requirement is not None
    pr = result.parsed_requirement
    assert pr.split_ratios.get("train") == pytest.approx(0.8)
    assert pr.stratify_by == "rhythm"
    # Custom split should be applied to user_options
    assert result.user_options.custom_split is not None
    assert result.user_options.custom_split["train"] == pytest.approx(0.8)


# ── Code tools ────────────────────────────────────────────────────────────

def test_write_script(tmp_path):
    from cardiomas.tools.code_tools import write_script

    script_path = str(tmp_path / "test_script.py")
    result = write_script.invoke({"script_path": script_path, "content": "print('hello')"})
    assert result["status"] == "ok"
    assert Path(script_path).exists()
    assert result["sha256"]


def test_execute_script_success(tmp_path):
    from cardiomas.tools.code_tools import execute_script

    script = tmp_path / "hello.py"
    script.write_text("print('hello world')")
    result = execute_script.invoke({"script_path": str(script), "timeout": 10})
    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]


def test_execute_script_failure(tmp_path):
    from cardiomas.tools.code_tools import execute_script

    script = tmp_path / "bad.py"
    script.write_text("raise RuntimeError('intentional')")
    result = execute_script.invoke({"script_path": str(script), "timeout": 10})
    assert result["exit_code"] != 0


def test_execute_script_not_found(tmp_path):
    from cardiomas.tools.code_tools import execute_script

    result = execute_script.invoke({"script_path": str(tmp_path / "nonexistent.py"), "timeout": 5})
    assert result["status"] == "error"


def test_verify_script_output_match(tmp_path):
    from cardiomas.tools.code_tools import verify_script_output

    splits = {"splits": {"train": ["r1", "r2"], "test": ["r3"]}}
    out_file = tmp_path / "out.json"
    ref_file = tmp_path / "ref.json"
    out_file.write_text(json.dumps(splits))
    ref_file.write_text(json.dumps(splits))

    result = verify_script_output.invoke({
        "output_splits_file": str(out_file),
        "reference_splits_file": str(ref_file),
    })
    assert result["match"] is True


def test_verify_script_output_mismatch(tmp_path):
    from cardiomas.tools.code_tools import verify_script_output

    out_file = tmp_path / "out.json"
    ref_file = tmp_path / "ref.json"
    out_file.write_text(json.dumps({"splits": {"train": ["r1"], "test": ["r2"]}}))
    ref_file.write_text(json.dumps({"splits": {"train": ["r1", "r3"], "test": ["r2"]}}))

    result = verify_script_output.invoke({
        "output_splits_file": str(out_file),
        "reference_splits_file": str(ref_file),
    })
    assert result["match"] is False
    assert len(result["mismatches"]) > 0


# ── Coder agent (mock LLM) ────────────────────────────────────────────────

def test_coder_agent_skips_without_splits():
    from cardiomas.agents.coder import coder_agent
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.recorder import SessionRecorder

    SessionRecorder.reset().start_session("test", {})
    state = GraphState(
        dataset_source="/data/test",
        user_options=UserOptions(dataset_source="/data/test"),
    )
    result = coder_agent(state)
    assert result.generated_scripts == {}


def test_coder_agent_generates_scripts(tmp_path):
    from cardiomas.agents.coder import coder_agent
    from cardiomas.schemas.state import GraphState, UserOptions
    from cardiomas.schemas.split import SplitManifest, ReproducibilityConfig
    from cardiomas.recorder import SessionRecorder
    from datetime import datetime

    SessionRecorder.reset().start_session("ptb-xl", {})

    repro = ReproducibilityConfig(
        cardiomas_version="0.2.0",
        seed=42,
        dataset_name="ptb-xl",
        dataset_source_url=None,
        dataset_checksum="abc123",
        split_strategy="deterministic",
        split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        timestamp=datetime.utcnow(),
    )
    manifest = SplitManifest(
        dataset_name="ptb-xl",
        reproducibility_config=repro,
        splits={"train": [f"r{i}" for i in range(70)],
                "val": [f"r{i}" for i in range(70, 85)],
                "test": [f"r{i}" for i in range(85, 100)]},
    )

    state = GraphState(
        dataset_source="/data/ptb-xl",
        user_options=UserOptions(
            dataset_source="/data/ptb-xl",
            output_dir=str(tmp_path),
        ),
        local_output_dir=str(tmp_path / "ptb-xl"),
        proposed_splits=manifest,
    )
    (tmp_path / "ptb-xl").mkdir()

    # Mock LLM and execute_script
    gen_script_code = "print('SPLITS_SHA256=abc123')\nprint('train: 70 records')"

    mock_exec = MagicMock()
    mock_exec.invoke.return_value = {
        "exit_code": 0,
        "stdout": "SPLITS_SHA256=abc123\ntrain: 70 records",
        "stderr": "",
        "status": "ok",
    }

    # Patch run_agent at source module + execute_script at tool module
    with patch("cardiomas.agents.base.run_agent", return_value=gen_script_code), \
         patch("cardiomas.tools.code_tools.execute_script", mock_exec):
        result = coder_agent(state)

    assert "generate_splits.py" in result.generated_scripts
    assert "verify_splits.py" in result.generated_scripts
    assert "explore_dataset.py" in result.generated_scripts
    assert Path(result.generated_scripts["generate_splits.py"]).exists()
    assert Path(result.generated_scripts["verify_splits.py"]).exists()


# ── Workflow hub-and-spoke ────────────────────────────────────────────────

def test_workflow_routing_table():
    """Verify the routing table covers all valid target names."""
    from cardiomas.graph.workflow import _ALL_TARGETS, _WORKER_AGENTS

    assert "end_saved" in _ALL_TARGETS
    assert "end_with_error" in _ALL_TARGETS
    assert "return_existing" in _ALL_TARGETS
    assert all(a in _ALL_TARGETS for a in _WORKER_AGENTS)


# ── Context compression helper ────────────────────────────────────────────

def test_compress_context_short():
    from cardiomas.agents.base import _compress_context

    short = "hello world"
    result, was_compressed, original_len = _compress_context(short, "test_agent")
    assert was_compressed is False
    assert result == short
    assert original_len == len(short)


def test_compress_context_long():
    from cardiomas.agents.base import _compress_context
    import cardiomas.config as cfg

    long_ctx = "word " * (cfg.CONTEXT_COMPRESS_THRESHOLD + 100)

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "compressed summary"
    mock_llm.invoke.return_value = mock_resp

    with patch("cardiomas.llm_factory.get_local_llm", return_value=mock_llm):
        result, was_compressed, original_len = _compress_context(long_ctx, "test_agent")

    assert was_compressed is True
    assert original_len > cfg.CONTEXT_COMPRESS_THRESHOLD


# ── verbose.py model name ────────────────────────────────────────────────

def test_vprint_llm_with_model_name(capsys):
    from cardiomas.verbose import enable, vprint_llm

    enable(True)
    # Should not raise with model_name parameter
    vprint_llm("analysis", "prompt text", "response text", model_name="gemma3:4b")
    enable(False)


# ── CLI resume command ────────────────────────────────────────────────────

def test_cli_resume_missing_file():
    from typer.testing import CliRunner
    from cardiomas.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["resume", "/tmp/nonexistent_checkpoint.json"])
    assert result.exit_code != 0
