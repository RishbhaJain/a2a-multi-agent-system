"""Regression tests for credential handling and committed secret material."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "GitHub token": re.compile(r"gh[pousr]_[0-9A-Za-z]{36,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_text_files() -> list[Path]:
    """Return text-like files tracked in the current repository."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        REPOSITORY_ROOT / relative_path
        for relative_path in completed.stdout.decode().split("\0")
        if relative_path and Path(relative_path).suffix.lower() in TEXT_SUFFIXES
    ]


class SecretHygieneTests(unittest.TestCase):
    def test_tracked_source_contains_no_known_secret_formats(self) -> None:
        findings: list[str] = []
        for path in tracked_text_files():
            contents = path.read_text(encoding="utf-8")
            for secret_kind, pattern in SECRET_PATTERNS.items():
                if pattern.search(contents):
                    findings.append(f"{path.relative_to(REPOSITORY_ROOT)}: {secret_kind}")

        self.assertEqual(
            findings,
            [],
            "credential-like values found in tracked files:\n" + "\n".join(findings),
        )

    def test_code_agent_fails_closed_without_gemini_key(self) -> None:
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.generativeai")
        fake_genai.configure = Mock()
        fake_genai.GenerativeModel = Mock()
        fake_google.generativeai = fake_genai

        spec = importlib.util.spec_from_file_location(
            "code_agent_under_test",
            REPOSITORY_ROOT / "a2a_simple" / "code_agent.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        with (
            patch.dict(
                sys.modules,
                {"google": fake_google, "google.generativeai": fake_genai},
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                module.CodeGenerationAgent()

        fake_genai.configure.assert_not_called()
        fake_genai.GenerativeModel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
