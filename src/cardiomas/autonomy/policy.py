from __future__ import annotations

import ast

from cardiomas.schemas.config import RuntimeConfig


_BANNED_MODULES = {"subprocess", "socket", "requests", "httpx", "urllib", "pip"}
_BANNED_CALLS = {"eval", "exec", "compile", "__import__", "input"}
_BANNED_SCRIPT_TOKENS = {" rm ", "sudo ", "pip install", "conda install", "curl ", "wget "}


def autonomy_enabled(config: RuntimeConfig) -> bool:
    return config.autonomy.enabled


def tool_codegen_allowed(config: RuntimeConfig) -> bool:
    return autonomy_enabled(config) and config.autonomy.allow_tool_codegen


def script_codegen_allowed(config: RuntimeConfig) -> bool:
    return autonomy_enabled(config) and config.autonomy.allow_script_codegen


def verify_python_ast(source: str, config: RuntimeConfig) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(source)
    allowed_modules = set(config.autonomy.allowed_python_modules)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name, allowed_modules, errors)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            _check_module(module, allowed_modules, errors)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _BANNED_CALLS:
                errors.append(f"Banned call used in generated tool: {name}")

    return errors


def verify_script_text(script: str) -> list[str]:
    lowered = f" {script.lower()} "
    return [f"Banned shell pattern detected: {token.strip()}" for token in _BANNED_SCRIPT_TOKENS if token in lowered]


def _check_module(module: str, allowed_modules: set[str], errors: list[str]) -> None:
    if not module:
        return
    root = module.split(".", 1)[0]
    if root == "__future__":
        return
    if root in _BANNED_MODULES:
        errors.append(f"Banned module imported in generated tool: {root}")
        return
    if root.startswith("cardiomas"):
        return
    if root not in allowed_modules:
        errors.append(f"Module not allowed in generated tool: {root}")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""
