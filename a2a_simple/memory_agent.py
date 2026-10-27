import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryPersistenceError(RuntimeError):
    """Raised when persistent memory cannot be loaded or committed safely."""


class MemoryAgent:
    """Agent that stores and retrieves information in persistent memory."""

    def __init__(self, memory_file: str = "memory_data.json"):
        self.memory_file = Path(memory_file)
        self._lock = threading.RLock()
        self.memory_data = self._load_memory()

    @staticmethod
    def _validate_memory(data: Any) -> dict[str, list[dict[str, str]]]:
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise MemoryPersistenceError("Memory file must contain an entries list")

        required_fields = {"id", "key", "value", "timestamp"}
        for entry in data["entries"]:
            if not isinstance(entry, dict) or not required_fields.issubset(entry):
                raise MemoryPersistenceError("Memory file contains an invalid entry")
            if not all(isinstance(entry[field], str) for field in required_fields):
                raise MemoryPersistenceError("Memory entry fields must be strings")
        return data

    def _load_memory(self) -> dict[str, list[dict[str, str]]]:
        """Load and validate memory without replacing corrupt data with an empty store."""
        if not self.memory_file.exists():
            return {"entries": []}

        try:
            with self.memory_file.open("r", encoding="utf-8") as memory_stream:
                return self._validate_memory(json.load(memory_stream))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryPersistenceError(
                f"Could not load memory from {self.memory_file}"
            ) from exc

    def _save_memory(self, memory_data: dict[str, list[dict[str, str]]]) -> None:
        """Atomically replace the memory file after flushing the complete JSON payload."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.memory_file.parent,
                prefix=f".{self.memory_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_stream:
                temporary_path = Path(temporary_stream.name)
                json.dump(memory_data, temporary_stream, indent=2)
                temporary_stream.write("\n")
                temporary_stream.flush()
                os.fsync(temporary_stream.fileno())
            os.replace(temporary_path, self.memory_file)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise MemoryPersistenceError(
                f"Could not save memory to {self.memory_file}"
            ) from exc

    def store(self, key: str, value: str) -> str:
        """Store a key-value pair only after its atomic commit succeeds."""
        try:
            with self._lock:
                entry = {
                    "id": str(len(self.memory_data["entries"]) + 1),
                    "key": key,
                    "value": value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                updated_memory = {
                    "entries": [*self.memory_data["entries"], entry]
                }
                self._save_memory(updated_memory)
                self.memory_data = updated_memory
            return f"Stored: {key} = {value}"

        except MemoryPersistenceError as exc:
            return f"Error storing memory: {exc}"

    def retrieve(self, search_term: str) -> str:
        """Retrieve information from the current in-process snapshot."""
        with self._lock:
            for entry in self.memory_data["entries"]:
                if search_term in entry["key"] or search_term in entry["value"]:
                    return f"Found: {entry['key']} = {entry['value']}"

        return f"Not found: {search_term}"

    def clear(self) -> str:
        """Clear memory only after the empty snapshot is committed."""
        try:
            with self._lock:
                empty_memory: dict[str, list[dict[str, str]]] = {"entries": []}
                self._save_memory(empty_memory)
                self.memory_data = empty_memory
            return "Memory cleared successfully"
        except MemoryPersistenceError as exc:
            return f"Error clearing memory: {exc}"
