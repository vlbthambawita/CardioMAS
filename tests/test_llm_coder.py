from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cardiomas.coding.llm_coder import (
    _extract_python_code,
    _format_dataset_context,
    discover_dataset_structure,
    synthesize_script,
)
from cardiomas.inference.base import ChatResponse
from cardiomas.schemas.config import LLMConfig, RuntimeConfig


# ---------------------------------------------------------------------------
# discover_dataset_structure
# ---------------------------------------------------------------------------

def test_discover_dataset_structure_csv(tmp_path):
    csv_file = tmp_path / "metadata.csv"
    csv_file.write_text("patient_id,age,label\n1,45,NORM\n2,60,AFIB\n", encoding="utf-8")
    (tmp_path / "signals").mkdir()
    (tmp_path / "signals" / "rec1.hea").write_text("dummy", encoding="utf-8")

    ctx = discover_dataset_structure(str(tmp_path))

    assert ctx["total_files"] == 2
    assert len(ctx["tabular_files"]) == 1
    tf = ctx["tabular_files"][0]
    assert tf["columns"] == ["patient_id", "age", "label"]
    assert tf["row_count"] == 2
    assert "metadata.csv" in tf["path"]
    assert "metadata.csv" in ctx["sample_files"][0] or any("metadata.csv" in p for p in ctx["sample_files"])


def test_discover_dataset_structure_tsv(tmp_path):
    tsv_file = tmp_path / "data.tsv"
    tsv_file.write_text("id\tvalue\n1\t2.5\n", encoding="utf-8")

    ctx = discover_dataset_structure(str(tmp_path))
    assert ctx["tabular_files"][0]["columns"] == ["id", "value"]


def test_discover_dataset_structure_empty_dir(tmp_path):
    ctx = discover_dataset_structure(str(tmp_path))
    assert ctx["total_files"] == 0
    assert ctx["tabular_files"] == []
    assert ctx["sample_files"] == []


def test_discover_dataset_structure_nonexistent():
    ctx = discover_dataset_structure("/nonexistent/path/that/does/not/exist")
    assert ctx["total_files"] == 0
    assert ctx["tabular_files"] == []


def test_discover_dataset_structure_caps_tabular_files(tmp_path):
    for i in range(10):
        f = tmp_path / f"file_{i}.csv"
        f.write_text(f"col_a,col_b\nv1,v2\n", encoding="utf-8")

    ctx = discover_dataset_structure(str(tmp_path))
    assert len(ctx["tabular_files"]) == 5  # capped at _MAX_TABULAR_FILES


# ---------------------------------------------------------------------------
# _extract_python_code
# ---------------------------------------------------------------------------

def test_extract_python_code_no_fences():
    code = "def main():\n    pass\n"
    assert _extract_python_code(code) == code


def test_extract_python_code_strips_python_fence():
    raw = "```python\ndef main():\n    pass\n```"
    result = _extract_python_code(raw)
    assert "```" not in result
    assert "def main():" in result


def test_extract_python_code_strips_plain_fence():
    raw = "```\nprint('hello')\n```"
    result = _extract_python_code(raw)
    assert "```" not in result
    assert "print('hello')" in result


def test_extract_python_code_adds_trailing_newline():
    raw = "print(1)"
    result = _extract_python_code(raw)
    assert result.endswith("\n")


# ---------------------------------------------------------------------------
# synthesize_script (mocked LLM)
# ---------------------------------------------------------------------------

_STUB_SCRIPT = (
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "DATASET_PATH = '/tmp/data'\n"
    "OUTPUT_DIR = Path('/tmp/out')\n"
    "\n"
    "def main():\n"
    "    result = {'ok': True, 'answer': 42}\n"
    "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "    (OUTPUT_DIR / 'results.json').write_text(json.dumps(result))\n"
    "    print(json.dumps(result))\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


def _make_chat_client(response_content: str) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = ChatResponse(
        model="test-model",
        content=response_content,
        raw={},
    )
    return client


def _make_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        scripts_dir=str(tmp_path / "scripts"),
        llm=LLMConfig(
            model="test-model",
            code_max_tokens=4000,
            code_temperature=0.2,
        ),
    )


def test_synthesize_script_calls_llm_twice(tmp_path):
    """First attempt: plan call + synthesis call = 2 LLM calls."""
    config = _make_config(tmp_path)
    client = _make_chat_client(_STUB_SCRIPT)

    result = synthesize_script(
        task="How many unique patients?",
        dataset_path=str(tmp_path),
        output_dir=str(tmp_path / "scripts" / "outputs"),
        config=config,
        chat_client=client,
    )

    assert client.chat.call_count == 2
    assert "def main" in result or "DATASET_PATH" in result


def test_synthesize_script_repair_skips_plan(tmp_path):
    """Repair path: only 1 LLM call (no re-planning)."""
    config = _make_config(tmp_path)
    client = _make_chat_client(_STUB_SCRIPT)

    result = synthesize_script(
        task="How many unique patients?",
        dataset_path=str(tmp_path),
        output_dir=str(tmp_path / "scripts" / "outputs"),
        config=config,
        chat_client=client,
        last_error="NameError: name 'df' is not defined",
        previous_code="import pandas as pd\ndf = pd.read_csv('x.csv')\nprint(len(df))\n",
    )

    assert client.chat.call_count == 1
    # The repair prompt must include both the error and previous code
    call_messages = client.chat.call_args[0][0].messages
    user_msg = next(m for m in call_messages if m.role == "user")
    assert "NameError" in user_msg.content
    assert "previous_code" in user_msg.content or "import pandas" in user_msg.content


def test_synthesize_script_repair_prompt_contains_error_and_code(tmp_path):
    """Verify the repair user message contains both error and previous code."""
    config = _make_config(tmp_path)
    client = _make_chat_client(_STUB_SCRIPT)

    synthesize_script(
        task="Count rows",
        dataset_path=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        config=config,
        chat_client=client,
        last_error="FileNotFoundError: data.csv not found",
        previous_code="open('data.csv')\n",
    )

    call_args = client.chat.call_args[0][0]
    user_content = next(m.content for m in call_args.messages if m.role == "user")
    assert "FileNotFoundError" in user_content
    assert "open('data.csv')" in user_content


def test_synthesize_script_strips_fences_from_llm_output(tmp_path):
    """LLM wraps code in markdown fences — they should be stripped."""
    config = _make_config(tmp_path)
    fenced = f"```python\n{_STUB_SCRIPT}```"
    client = _make_chat_client(fenced)

    result = synthesize_script(
        task="count rows",
        dataset_path=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        config=config,
        chat_client=client,
    )

    assert "```" not in result
    assert "def main" in result


def test_synthesize_script_uses_code_max_tokens(tmp_path):
    """ChatRequest sent to LLM uses code_max_tokens, not default 800."""
    config = _make_config(tmp_path)
    client = _make_chat_client(_STUB_SCRIPT)

    synthesize_script(
        task="list labels",
        dataset_path=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        config=config,
        chat_client=client,
    )

    # Second call is code synthesis — check its max_tokens
    synthesis_call_request = client.chat.call_args_list[-1][0][0]
    assert synthesis_call_request.max_tokens == 4000


# ---------------------------------------------------------------------------
# Fallback: no chat_client → build_standalone_script uses template
# ---------------------------------------------------------------------------

def test_build_standalone_script_template_fallback(tmp_path):
    """When chat_client=None, build_standalone_script falls back to template."""
    from cardiomas.coding.tool_builder import build_standalone_script
    from cardiomas.schemas.config import RuntimeConfig

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        scripts_dir=str(tmp_path / "scripts"),
    )
    script = build_standalone_script(
        task="How many unique patients?",
        dataset_path=str(tmp_path),
        config=config,
        chat_client=None,
    )

    assert script.script_name.endswith(".py")
    assert "def main()" in script.code
    assert "DATASET_PATH" in script.code


def test_build_standalone_script_uses_llm_when_client_provided(tmp_path):
    """When chat_client is provided, LLM is called (not the template)."""
    from cardiomas.coding.tool_builder import build_standalone_script

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        scripts_dir=str(tmp_path / "scripts"),
        llm=LLMConfig(model="test-model", code_max_tokens=2000),
    )
    client = _make_chat_client(_STUB_SCRIPT)

    script = build_standalone_script(
        task="How many unique patients?",
        dataset_path=str(tmp_path),
        config=config,
        chat_client=client,
    )

    assert client.chat.called
    assert script.code == _STUB_SCRIPT


# ---------------------------------------------------------------------------
# AutonomousToolManager: chat_client threading
# ---------------------------------------------------------------------------

def test_autonomous_tool_manager_stores_chat_client(tmp_path):
    from cardiomas.autonomy.recovery import AutonomousToolManager

    config = RuntimeConfig(output_dir=str(tmp_path / "output"))
    client = _make_chat_client(_STUB_SCRIPT)
    manager = AutonomousToolManager(config, chat_client=client)

    assert manager.chat_client is client


def test_autonomous_tool_manager_default_no_client(tmp_path):
    from cardiomas.autonomy.recovery import AutonomousToolManager

    config = RuntimeConfig(output_dir=str(tmp_path / "output"))
    manager = AutonomousToolManager(config)

    assert manager.chat_client is None
