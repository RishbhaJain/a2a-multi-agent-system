"""Evaluate when the router should act and when it should abstain."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evals.evaluate_router import load_scenarios
from routing_policy import ROUTE_PRIORITY, classify_question


EXPECTED_KINDS = {"in_domain", "out_of_domain", "ambiguous"}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _validate_scenarios(scenarios: list[dict[str, object]]) -> None:
    if not scenarios:
        raise ValueError("selective-routing dataset must not be empty")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    for scenario in scenarios:
        scenario_id = str(scenario.get("id", ""))
        kind = str(scenario.get("kind", ""))
        expected_tool = str(scenario.get("expected_tool", ""))
        prompt = str(scenario.get("prompt", ""))
        if not scenario_id or scenario_id in ids:
            raise ValueError(f"scenario id must be unique and non-empty: {scenario_id!r}")
        if kind not in EXPECTED_KINDS:
            raise ValueError(f"unknown scenario kind for {scenario_id}: {kind!r}")
        if not prompt.strip():
            raise ValueError(f"prompt must be non-empty for {scenario_id}")
        if kind == "in_domain" and expected_tool not in ROUTE_PRIORITY:
            raise ValueError(f"in-domain scenario {scenario_id} needs a known tool")
        if kind != "in_domain" and expected_tool != "unknown":
            raise ValueError(f"abstention scenario {scenario_id} must expect unknown")
        ids.add(scenario_id)
        counts[kind] += 1
    missing = EXPECTED_KINDS - counts.keys()
    if missing:
        raise ValueError(f"dataset is missing scenario kinds: {sorted(missing)}")


def evaluate(dataset_path: Path) -> dict[str, object]:
    scenarios = load_scenarios(dataset_path)
    _validate_scenarios(scenarios)

    rows = []
    kind_counts: Counter[str] = Counter()
    kind_correct: Counter[str] = Counter()
    in_domain_routed = 0
    in_domain_correct = 0
    abstention_total = 0
    abstention_correct = 0
    false_routes = 0

    for scenario in scenarios:
        decision = classify_question(str(scenario["prompt"]))
        expected_tool = str(scenario["expected_tool"])
        kind = str(scenario["kind"])
        correct = decision.tool == expected_tool
        kind_counts[kind] += 1
        kind_correct[kind] += int(correct)

        if kind == "in_domain":
            in_domain_routed += int(decision.tool != "unknown")
            in_domain_correct += int(correct)
        else:
            abstention_total += 1
            abstention_correct += int(decision.tool == "unknown")
            false_routes += int(decision.tool != "unknown")

        rows.append(
            {
                "id": scenario["id"],
                "kind": kind,
                "expected_tool": expected_tool,
                "predicted_tool": decision.tool,
                "score_margin": decision.score_margin,
                "abstention_reason": decision.abstention_reason,
                "scores": decision.scores,
                "correct": correct,
            }
        )

    total = len(scenarios)
    metrics = {
        "scenario_count": total,
        "routing_accuracy": _ratio(sum(kind_correct.values()), total),
        "in_domain_coverage": _ratio(in_domain_routed, kind_counts["in_domain"]),
        "in_domain_accuracy": _ratio(in_domain_correct, kind_counts["in_domain"]),
        "abstention_recall": _ratio(abstention_correct, abstention_total),
        "false_route_rate": _ratio(false_routes, abstention_total),
        "by_kind": {
            kind: {
                "count": kind_counts[kind],
                "accuracy": _ratio(kind_correct[kind], kind_counts[kind]),
            }
            for kind in sorted(EXPECTED_KINDS)
        },
    }
    return {
        "schema_version": 1,
        "dataset": dataset_path.name,
        "policy": "margin_abstention_v1",
        "metrics": metrics,
        "failures": [row for row in rows if not row["correct"]],
        "decisions": rows,
        "regression_gate": {
            "in_domain_coverage_min": 1.0,
            "in_domain_accuracy_min": 1.0,
            "abstention_recall_min": 1.0,
            "false_route_rate_max": 0.0,
        },
    }


def check_regression(result: dict[str, object], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = result["metrics"]
    prior = baseline["metrics"]
    gate = result["regression_gate"]
    errors = []

    for metric in (
        "in_domain_coverage",
        "in_domain_accuracy",
        "abstention_recall",
    ):
        minimum = gate[f"{metric}_min"]
        if current[metric] < minimum:
            errors.append(f"{metric} {current[metric]} is below {minimum}")
        if current[metric] < prior[metric]:
            errors.append(f"{metric} regressed from {prior[metric]} to {current[metric]}")
    maximum = gate["false_route_rate_max"]
    if current["false_route_rate"] > maximum:
        errors.append(f"false_route_rate {current['false_route_rate']} exceeds {maximum}")
    if current["false_route_rate"] > prior["false_route_rate"]:
        errors.append(
            "false_route_rate regressed from "
            f"{prior['false_route_rate']} to {current['false_route_rate']}"
        )
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
