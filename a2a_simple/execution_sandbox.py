"""Constrained subprocess runner for model-generated Python programs.

This module provides defense in depth for the code agent. It is not a replacement
for a container or microVM when executing adversarial, untrusted workloads.
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on Linux/macOS.
    resource = None


ALLOWED_IMPORTS = frozenset(
    {
        "bisect",
        "cmath",
        "collections",
        "decimal",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "math",
        "random",
        "statistics",
        "string",
    }
)
BLOCKED_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)
BLOCKED_NAMES = frozenset(
    {
        "builtins",
        "ctypes",
        "importlib",
        "marshal",
        "os",
        "pathlib",
        "pickle",
        "shutil",
        "socket",
        "subprocess",
        "sys",
    }
)


class UnsafeCodeError(ValueError):
    """Raised when generated code violates the execution policy."""


@dataclass(frozen=True)
class ExecutionResult:
    """Structured result from the constrained Python subprocess."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class _SafetyVisitor(ast.NodeVisitor):
    def _reject(self, node: ast.AST, reason: str) -> None:
        line = getattr(node, "lineno", "unknown")
        raise UnsafeCodeError(f"{reason} at line {line}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", maxsplit=1)[0]
            if root not in ALLOWED_IMPORTS:
                self._reject(node, f"import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", maxsplit=1)[0]
        if node.level or root not in ALLOWED_IMPORTS:
            self._reject(node, f"import from '{node.module}' is not allowed")
        if any(alias.name == "*" for alias in node.names):
            self._reject(node, "wildcard imports are not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BLOCKED_NAMES:
            self._reject(node, f"name '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self._reject(node, "private and dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            self._reject(node, f"call to '{node.func.id}' is not allowed")
        self.generic_visit(node)


def validate_code(code: str, *, max_source_bytes: int = 20_000) -> None:
    """Reject source that exceeds the documented code-execution policy."""
    if len(code.encode("utf-8")) > max_source_bytes:
        raise UnsafeCodeError(f"source exceeds the {max_source_bytes}-byte limit")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"invalid Python syntax at line {exc.lineno}") from exc
    _SafetyVisitor().visit(tree)


def _resource_limiter(
    timeout_seconds: float, memory_limit_mb: int
) -> Callable[[], None] | None:
    if resource is None:
        return None

    def set_limit(kind: int, soft: int, hard: int | None = None) -> None:
        _, current_hard = resource.getrlimit(kind)
        requested_hard = soft if hard is None else hard
        if current_hard != resource.RLIM_INFINITY:
            requested_hard = min(requested_hard, current_hard)
        requested_soft = min(soft, requested_hard)
        resource.setrlimit(kind, (requested_soft, requested_hard))

    def apply_limits() -> None:
        cpu_seconds = max(1, math.ceil(timeout_seconds))
        memory_bytes = memory_limit_mb * 1024 * 1024
        set_limit(resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1)
        set_limit(resource.RLIMIT_AS, memory_bytes)
        set_limit(resource.RLIMIT_FSIZE, 1024 * 1024)
        set_limit(resource.RLIMIT_NOFILE, 16)

    return apply_limits


def _read_limited(path: Path, max_bytes: int) -> tuple[str, bool]:
    size = path.stat().st_size
    content = path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return content, size > max_bytes


def execute_python(
    code: str,
    *,
    timeout_seconds: float = 5,
    memory_limit_mb: int = 256,
    max_output_bytes: int = 64 * 1024,
) -> ExecutionResult:
    """Validate and execute Python with isolation flags and resource limits."""
    validate_code(code)

    with tempfile.TemporaryDirectory(prefix="a2a-code-") as temp_dir:
        directory = Path(temp_dir)
        script_path = directory / "program.py"
        stdout_path = directory / "stdout.txt"
        stderr_path = directory / "stderr.txt"
        script_path.write_text(code, encoding="utf-8")

        environment = {
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
        timed_out = False
        returncode: int | None = None

        with (
            stdout_path.open("wb") as stdout_file,
            stderr_path.open("wb") as stderr_file,
        ):
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-S", str(script_path)],
                    cwd=directory,
                    env=environment,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout_seconds,
                    check=False,
                    preexec_fn=_resource_limiter(timeout_seconds, memory_limit_mb),
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True

        stdout, stdout_truncated = _read_limited(stdout_path, max_output_bytes)
        stderr, stderr_truncated = _read_limited(stderr_path, max_output_bytes)
        return ExecutionResult(
            returncode=returncode,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            timed_out=timed_out,
            output_truncated=stdout_truncated or stderr_truncated,
        )
