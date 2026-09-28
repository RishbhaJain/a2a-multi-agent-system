"""Benchmark the constrained code-execution boundary without external services."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from execution_sandbox import UnsafeCodeError, execute_python


OUTCOMES = frozenset(
    {"success", "policy_denied", "runtime_error", "timeout", "output_truncated"}
)


def load_scenarios(path: Path) -> list[dict[str, object]]:
    """Load and validate execution scenarios from JSON Lines."""
    scenarios: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            scenario = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc

        missing = {"id", "code", "expected_outcome"} - scenario.keys()
        if missing:
            raise ValueError(
                f"scenario on line {line_number} is missing: {', '.join(sorted(missing))}"
            )
        scenario_id = str(scenario["id"])
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)
        if scenario["expected_outcome"] not in OUTCOMES:
            raise ValueError(
                f"scenario {scenario_id} has unknown outcome: "
                f"{scenario['expected_outcome']}"
            )
        scenarios.append(scenario)

    if not scenarios:
        raise ValueError("execution dataset is empty")
    return scenarios


def _run_scenario(scenario: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    returncode: int | None = None
    timed_out = False
    output_truncated = False
    stdout = ""
    try:
        result = execute_python(
            str(scenario["code"]),
            timeout_seconds=float(scenario.get("timeout_seconds", 1.0)),
            memory_limit_mb=int(scenario.get("memory_limit_mb", 256)),
            max_output_bytes=int(scenario.get("max_output_bytes", 64 * 1024)),
        )
        returncode = result.returncode
        timed_out = result.timed_out
        output_truncated = result.output_truncated
        stdout = result.stdout
        if result.timed_out:
            outcome = "timeout"
        elif result.output_truncated:
            outcome = "output_truncated"
        elif result.succeeded:
            outcome = "success"
        else:
            outcome = "runtime_error"
    except UnsafeCodeError:
        outcome = "policy_denied"

    expected_outcome = str(scenario["expected_outcome"])
    expected_stdout = scenario.get("expected_stdout")
    outcome_ok = outcome == expected_outcome
    stdout_ok = expected_stdout is None or stdout == expected_stdout
    failures = []
    if not outcome_ok:
        failures.append("outcome_mismatch")
    if not stdout_ok:
        failures.append("stdout_mismatch")

    return {
        "id": scenario["id"],
        "expected_outcome": expected_outcome,
        "actual_outcome": outcome,
        "success": outcome_ok and stdout_ok,
        "failure_categories": failures,
        "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
        "returncode": returncode,
        "timed_out": timed_out,
        "output_truncated": output_truncated,
    }


def evaluate_scenarios(scenarios: list[dict[str, object]]) -> dict[str, object]:
    """Execute scenarios and aggregate reliability and latency metrics."""
    if not scenarios:
        raise ValueError("execution scenarios are empty")

    rows = [_run_scenario(scenario) for scenario in scenarios]
    counts = Counter(str(scenario["expected_outcome"]) for scenario in scenarios)
    successes = Counter(
        row["expected_outcome"] for row in rows if bool(row["success"])
    )
    failures = Counter(
        category for row in rows for category in row["failure_categories"]
    )
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)

    def rate(outcome: str) -> float:
        total = counts[outcome]
        return round(successes[outcome] / total, 4) if total else 1.0

    return {
        "scenario_count": len(rows),
        "outcome_accuracy": round(
            sum(bool(row["success"]) for row in rows) / len(rows), 4
        ),
        "safe_execution_success_rate": rate("success"),
        "policy_rejection_accuracy": rate("policy_denied"),
        "runtime_failure_detection_rate": rate("runtime_error"),
        "timeout_detection_rate": rate("timeout"),
        "truncation_detection_rate": rate("output_truncated"),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(latencies[p95_index], 3),
        },
        "failure_counts": dict(sorted(failures.items())),
        "rows": rows,
    }


def evaluate(dataset_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": dataset_path.name,
        "metrics": evaluate_scenarios(load_scenarios(dataset_path)),
        "regression_gate": {
            "outcome_accuracy_min": 1.0,
            "safe_execution_success_rate_min": 1.0,
            "policy_rejection_accuracy_min": 1.0,
            "runtime_failure_detection_rate_min": 1.0,
            "timeout_detection_rate_min": 1.0,
            "truncation_detection_rate_min": 1.0,
        },
    }


def check_regression(result: dict[str, object], baseline_path: Path) -> None:
    """Fail when a stable reliability metric misses its gate or baseline."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = result["metrics"]
    prior = baseline["metrics"]
    gate = result["regression_gate"]
    metric_names = (
        "outcome_accuracy",
        "safe_execution_success_rate",
        "policy_rejection_accuracy",
        "runtime_failure_detection_rate",
        "timeout_detection_rate",
        "truncation_detection_rate",
    )

    errors = []
    for metric in metric_names:
        minimum = gate[f"{metric}_min"]
        if current[metric] < minimum:
            errors.append(f"{metric} {current[metric]} is below {minimum}")
        if current[metric] < prior[metric]:
            errors.append(f"{metric} regressed from {prior[metric]} to {current[metric]}")
    if errors:
        raise SystemExit("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()

    result = evaluate(args.dataset)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.check:
        check_regression(result, args.check)


if __name__ == "__main__":
    main()
