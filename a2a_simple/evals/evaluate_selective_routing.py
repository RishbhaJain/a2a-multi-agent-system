"""Evaluate when the router should act and when it should abstain."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evals.evaluate_router import load_scenarios
from routing_policy import MIN_CONFIDENT_MARGIN, ROUTE_PRIORITY, classify_question


EXPECTED_KINDS = {"in_domain", "out_of_domain", "ambiguous"}
MARGIN_SWEEP = tuple(range(MIN_CONFIDENT_MARGIN + 1))


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


def _evaluate_scenarios(
    scenarios: list[dict[str, object]], *, min_confident_margin: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    kind_counts: Counter[str] = Counter()
    kind_correct: Counter[str] = Counter()
    in_domain_routed = 0
    in_domain_correct = 0
    abstention_total = 0
    abstention_correct = 0
    false_routes = 0
    routed_total = 0
    routed_correct = 0

    for scenario in scenarios:
        decision = classify_question(
            str(scenario["prompt"]),
            min_confident_margin=min_confident_margin,
        )
        expected_tool = str(scenario["expected_tool"])
        kind = str(scenario["kind"])
        correct = decision.tool == expected_tool
        routed = decision.tool != "unknown"
        kind_counts[kind] += 1
        kind_correct[kind] += int(correct)
        routed_total += int(routed)
        routed_correct += int(routed and correct)

        if kind == "in_domain":
            in_domain_routed += int(routed)
            in_domain_correct += int(correct)
        else:
            abstention_total += 1
            abstention_correct += int(decision.tool == "unknown")
            false_routes += int(routed)

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
        "routed_count": routed_total,
        "coverage": _ratio(routed_total, total),
        "selective_accuracy": _ratio(routed_correct, routed_total),
        "selective_risk": round(1 - _ratio(routed_correct, routed_total), 4),
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
    return metrics, rows


def _risk_coverage_curve(
    scenarios: list[dict[str, object]],
) -> dict[str, object]:
    operating_points = []
    for margin in MARGIN_SWEEP:
        metrics, _ = _evaluate_scenarios(
            scenarios,
            min_confident_margin=margin,
        )
        operating_points.append(
            {
                "min_confident_margin": margin,
                "routed_count": metrics["routed_count"],
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
                "selective_risk": metrics["selective_risk"],
                "abstention_recall": metrics["abstention_recall"],
                "false_route_rate": metrics["false_route_rate"],
            }
        )
    minimum_zero_false_route_margin = next(
        (
            point["min_confident_margin"]
            for point in operating_points
            if point["false_route_rate"] == 0.0
        ),
        None,
    )
    return {
        "swept_parameter": "min_confident_margin",
        "production_margin": MIN_CONFIDENT_MARGIN,
        "minimum_zero_false_route_margin": minimum_zero_false_route_margin,
        "operating_points": operating_points,
    }


def evaluate(dataset_path: Path) -> dict[str, object]:
    scenarios = load_scenarios(dataset_path)
    _validate_scenarios(scenarios)
    metrics, rows = _evaluate_scenarios(
        scenarios,
        min_confident_margin=MIN_CONFIDENT_MARGIN,
    )
    return {
        "schema_version": 2,
        "dataset": dataset_path.name,
        "policy": "margin_abstention_v1",
        "metrics": metrics,
        "risk_coverage_curve": _risk_coverage_curve(scenarios),
        "failures": [row for row in rows if not row["correct"]],
        "decisions": rows,
        "regression_gate": {
            "in_domain_coverage_min": 1.0,
            "in_domain_accuracy_min": 1.0,
            "abstention_recall_min": 1.0,
            "false_route_rate_max": 0.0,
            "minimum_zero_false_route_margin_max": MIN_CONFIDENT_MARGIN,
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
    curve = result["risk_coverage_curve"]
    zero_false_route_margin = curve["minimum_zero_false_route_margin"]
    maximum_margin = gate["minimum_zero_false_route_margin_max"]
    if zero_false_route_margin is None or zero_false_route_margin > maximum_margin:
        errors.append(
            "risk-coverage sweep does not reach zero false routes by "
            f"margin {maximum_margin}"
        )
    prior_curve = baseline.get("risk_coverage_curve")
    prior_zero_false_route_margin = (
        prior_curve.get("minimum_zero_false_route_margin") if prior_curve else None
    )
    if (
        zero_false_route_margin is not None
        and prior_zero_false_route_margin is not None
        and zero_false_route_margin > prior_zero_false_route_margin
    ):
        errors.append(
            "minimum zero-false-route margin regressed from "
            f"{prior_zero_false_route_margin} to "
            f"{zero_false_route_margin}"
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
