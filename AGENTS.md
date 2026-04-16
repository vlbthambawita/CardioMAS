# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/cardiomas/`. Use `agents/` for pipeline workers, `graph/` for LangGraph workflow wiring, `cli/main.py` for the Typer CLI, `tools/` for tool wrappers, `schemas/` for Pydantic models, and `splitters/` for deterministic split logic. Dataset metadata lives in `src/cardiomas/datasets/registry.yaml`, and agent prompt files live in `src/cardiomas/skills/`. Put new tests in `tests/`, usually in a matching `test_<area>.py` module. Treat `output/` and `dist/` as generated artifacts, not source.

## Build, Test, and Development Commands
Use `pip install -e ".[dev]"` for an editable install with test and lint dependencies. Run the CLI locally with commands such as `cardiomas analyze /data/ptb-xl/ --verbose` or `cardiomas status ptb-xl`. Match CI before opening a PR:

- `ruff check src/ tests/` runs lint and import-order checks.
- `ruff format src/ tests/` formats the codebase.
- `pytest tests/ -v --cov=cardiomas --cov-report=xml` runs the full suite with coverage output.
- `python -m build` builds the sdist and wheel; package versioning comes from git tags via `setuptools-scm`.

## Coding Style & Naming Conventions
Target Python 3.10+ and use 4-space indentation. Follow the Ruff config in `pyproject.toml`: max line length is 100, with lint rules for errors, imports, and pyupgrade. Prefer type hints on public functions and keep docstrings brief and behavioral. Use `snake_case` for modules and functions, `PascalCase` for classes and Pydantic models, and descriptive CLI option names such as `--force-reanalysis`.

## Testing Guidelines
Pytest is the test runner. Name files `test_<feature>.py`, test functions `test_<behavior>`, and group related cases in `Test...` classes where useful. Prefer deterministic fixtures and synthetic record IDs, as in `tests/test_splitters.py`. Mock external services for HuggingFace, Ollama, and publishing paths in CLI-facing tests. Run focused tests while iterating, for example `pytest tests/test_cli.py -v`.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit prefixes such as `feat:`, `fix:`, and `docs:`; keep messages imperative and specific, for example `fix: resolve v4 output dirs`. PRs should include a short summary, linked issue when relevant, and the exact validation commands you ran. Include CLI output or screenshots for user-visible changes. Never commit `.env`, API tokens, generated `output/` data, or raw patient data.
