import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_agent import MemoryAgent, MemoryPersistenceError


class MemoryAgentTests(unittest.TestCase):
    def test_concurrent_stores_preserve_every_entry_with_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            agent = MemoryAgent(str(memory_path))
            workers = [
                threading.Thread(target=agent.store, args=(f"key-{index}", "value"))
                for index in range(20)
            ]

            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            persisted = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(len(persisted["entries"]), 20)
            self.assertEqual(
                {entry["id"] for entry in persisted["entries"]},
                {str(index) for index in range(1, 21)},
            )
            self.assertEqual(
                {entry["key"] for entry in persisted["entries"]},
                {f"key-{index}" for index in range(20)},
            )

    def test_failed_atomic_replace_keeps_file_and_memory_snapshot_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            agent = MemoryAgent(str(memory_path))
            self.assertEqual(agent.store("stable", "value"), "Stored: stable = value")
            original_payload = memory_path.read_text(encoding="utf-8")

            with patch("memory_agent.os.replace", side_effect=OSError("disk failure")):
                result = agent.store("uncommitted", "value")

            self.assertIn("Error storing memory", result)
            self.assertEqual(memory_path.read_text(encoding="utf-8"), original_payload)
            self.assertEqual(len(agent.memory_data["entries"]), 1)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_corrupt_memory_fails_closed_instead_of_becoming_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            memory_path.write_text("{not valid json", encoding="utf-8")

            with self.assertRaisesRegex(MemoryPersistenceError, "Could not load"):
                MemoryAgent(str(memory_path))

            self.assertEqual(memory_path.read_text(encoding="utf-8"), "{not valid json")

    def test_clear_failure_preserves_committed_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            agent = MemoryAgent(str(memory_path))
            agent.store("stable", "value")

            with patch("memory_agent.os.replace", side_effect=OSError("disk failure")):
                result = agent.clear()

            self.assertIn("Error clearing memory", result)
            self.assertEqual(agent.retrieve("stable"), "Found: stable = value")
            persisted = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(len(persisted["entries"]), 1)

    def test_atomic_commit_uses_replace_in_same_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            agent = MemoryAgent(str(memory_path))
            real_replace = os.replace

            with patch("memory_agent.os.replace", wraps=real_replace) as replace:
                agent.store("key", "value")

            source, destination = replace.call_args.args
            self.assertEqual(Path(source).parent, memory_path.parent)
            self.assertEqual(destination, memory_path)


if __name__ == "__main__":
    unittest.main()
