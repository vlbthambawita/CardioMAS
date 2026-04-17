from __future__ import annotations

from cardiomas.schemas.config import RuntimeConfig


def build_shell_script(task: str, dataset_path: str, config: RuntimeConfig) -> str:
    lowered = task.lower()
    header = "#!/usr/bin/env bash\nset -eu\n\n"

    if "corpus" in lowered:
        return (
            header
            + f"CONFIG_PATH=${{1:-{config.output_dir}/runtime.yaml}}\n"
            + 'echo "Building CardioMAS corpus"\n'
            + 'cardiomas build-corpus --config "$CONFIG_PATH"\n'
        )

    if "metadata" in lowered or "scan" in lowered or "inspect" in lowered:
        target = dataset_path or "."
        return (
            header
            + f'DATASET_PATH="${{1:-{target}}}"\n'
            + 'echo "Scanning dataset files"\n'
            + 'find "$DATASET_PATH" -maxdepth 2 -type f | sort | head -n 200\n'
        )

    target = dataset_path or "."
    return (
        header
        + f'DATASET_PATH="${{1:-{target}}}"\n'
        + 'echo "CardioMAS generated task script"\n'
        + 'echo "Task: ' + _escape_for_double_quotes(task) + '"\n'
        + 'find "$DATASET_PATH" -type f | sort | head -n 100\n'
    )


def _escape_for_double_quotes(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
