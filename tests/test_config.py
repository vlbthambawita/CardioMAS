from __future__ import annotations

from pathlib import Path

import pytest

from cardiomas.schemas.config import RuntimeConfig


def test_runtime_config_resolves_paths_relative_to_config(tmp_path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    (source_dir / "notes.md").write_text("demo note", encoding="utf-8")
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_dir: output",
                "sources:",
                "  - kind: local_dir",
                "    path: data",
                "    label: demo",
            ]
        ),
        encoding="utf-8",
    )

    config = RuntimeConfig.from_file(str(config_path))

    assert config.output_dir == str((tmp_path / "output").resolve())
    assert config.sources[0].path == str(source_dir.resolve())


def test_runtime_config_reports_yaml_syntax_location(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "system_name: CardioMAS",
                "  output_dir: output",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Invalid YAML .*line 2, column 13"):
        RuntimeConfig.from_file(str(config_path))
