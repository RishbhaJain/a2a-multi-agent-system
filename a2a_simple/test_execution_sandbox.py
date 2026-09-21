import importlib
import sys
import types
import unittest
from unittest.mock import patch

from execution_sandbox import UnsafeCodeError, execute_python, validate_code


class CodePolicyTests(unittest.TestCase):
    def test_allows_deterministic_math_program(self):
        validate_code(
            "import math\nvalues = [math.factorial(n) for n in range(6)]\nprint(sum(values))"
        )

    def test_rejects_operating_system_import(self):
        with self.assertRaisesRegex(UnsafeCodeError, "import of 'os'"):
            validate_code("import os\nprint(os.getcwd())")

    def test_rejects_file_access(self):
        with self.assertRaisesRegex(UnsafeCodeError, "call to 'open'"):
            validate_code("print(open('/etc/passwd').read())")

    def test_rejects_dunder_traversal(self):
        with self.assertRaisesRegex(UnsafeCodeError, "dunder attribute"):
            validate_code("print((1).__class__.__mro__)")

    def test_rejects_invalid_syntax_before_execution(self):
        with self.assertRaisesRegex(UnsafeCodeError, "invalid Python syntax"):
            validate_code("for")


class ConstrainedExecutionTests(unittest.TestCase):
    def test_executes_valid_program_in_isolated_interpreter(self):
        result = execute_python("from math import factorial\nprint(factorial(5))")
        self.assertTrue(result.succeeded, result.stderr)
        self.assertEqual(result.stdout, "120")

    def test_times_out_non_terminating_program(self):
        result = execute_python("while True:\n    pass", timeout_seconds=0.2)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.succeeded)

    def test_caps_captured_output(self):
        result = execute_python("print('x' * 10000)", max_output_bytes=128)
        self.assertTrue(result.succeeded, result.stderr)
        self.assertEqual(len(result.stdout), 128)
        self.assertTrue(result.output_truncated)


class CodeAgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _load_agent_class():
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.generativeai")
        fake_google.generativeai = fake_genai
        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.generativeai": fake_genai},
        ):
            module = importlib.import_module("code_execution_agent")
        return module.CodeExecutionAgent

    async def test_agent_executes_policy_compliant_program(self):
        agent_class = self._load_agent_class()
        agent = agent_class.__new__(agent_class)
        self.assertEqual(await agent._execute_code("print(6 * 7)"), "42")

    async def test_agent_reports_policy_rejection(self):
        agent_class = self._load_agent_class()
        agent = agent_class.__new__(agent_class)
        result = await agent._execute_code("import os\nprint(os.getcwd())")
        self.assertIn("rejected by safety policy", result)


if __name__ == "__main__":
    unittest.main()
