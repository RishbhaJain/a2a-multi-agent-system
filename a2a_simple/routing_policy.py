"""Deterministic routing policy with inspectable scores and arguments."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    tool: str
    scores: dict[str, int]
    matched_rules: dict[str, list[str]]


ROUTE_PRIORITY = ("memory", "code", "web", "hash", "math")
RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "memory": (
        (r"\b(remember|store|save)\b", 5),
        (r"\b(recall|retrieve|memory|stored|saved)\b", 5),
        (r"\bpaired with\b", 6),
        (r"\bwhat was\b", 3),
    ),
    "code": (
        (r"\b(write|generate|create)\b.{0,40}\b(program|code|script|function)\b", 7),
        (r"\b(implement|algorithm|programming|python code)\b", 6),
        (r"\bcode\b", 4),
    ),
    "web": (
        (r"https?://", 8),
        (r"\b(go to|visit|browse|website|webpage|browser)\b", 6),
        (r"\b(search the web|look up online)\b", 6),
    ),
    "hash": (
        (r"\b(md5|sha(?:1|224|256|384|512)?|hash|digest|checksum)\b", 8),
        (r"\bcryptographic\b", 5),
    ),
    "math": (
        (r"\b(solve|calculate|derivative|integral|equation|arithmetic|algebra)\b", 5),
        (r"\b(area|perimeter|volume|average|factorial|mean|sum|product)\b", 3),
        (r"\d\s*[+\-*/=^]\s*\d", 6),
    ),
}


def classify_question(question: str) -> RoutingDecision:
    """Classify a request using weighted, auditable rules."""
    normalized = " ".join(question.lower().split())
    scores = {route: 0 for route in ROUTE_PRIORITY}
    matches = {route: [] for route in ROUTE_PRIORITY}

    for route, rules in RULES.items():
        for pattern, weight in rules:
            if re.search(pattern, normalized):
                scores[route] += weight
                matches[route].append(pattern)

    winning_score = max(scores.values(), default=0)
    if winning_score == 0:
        return RoutingDecision("unknown", scores, matches)

    winner = next(
        route for route in ROUTE_PRIORITY if scores[route] == winning_score
    )
    return RoutingDecision(winner, scores, matches)


def legacy_classify_question(question: str) -> str:
    """Preserve the original substring router as an evaluation baseline."""
    question_lower = question.lower()
    keywords = {
        "hash": [
            "hash",
            "md5",
            "sha512",
            "sha",
            "execute",
            "sequence",
            "operations",
            "cryptographic",
            "digest",
        ],
        "math": [
            "solve",
            "calculate",
            "what is",
            "math",
            "arithmetic",
            "algebra",
            "equation",
            "derivative",
            "integral",
            "area",
            "perimeter",
            "volume",
            "plus",
            "minus",
            "times",
            "divided",
        ],
        "web": [
            "go to",
            "visit",
            "browse",
            "website",
            "play",
            "game",
            "tic-tac-toe",
            "ttt.puppy9.com",
            "secret number",
            "congratulation",
            "win",
            "browser",
            "web",
            "url",
            "http",
            "https",
        ],
        "memory": [
            "remember",
            "store",
            "save",
            "recall",
            "retrieve",
            "memory",
            "previously",
            "earlier",
            "before",
            "past",
            "stored",
            "saved",
            "what was",
            "paired with",
            "check your memory",
            "tell me",
        ],
        "code": [
            "write a program",
            "program that",
            "compute",
            "algorithm",
            "prime numbers",
            "sum of squares",
            "modulo",
            "programming",
            "code",
            "execute",
            "generate code",
            "write code",
            "implement",
            "calculate",
            "complex",
            "mathematical problem",
            "step by step",
            "numerical result",
            "final result",
            "output the result",
        ],
    }
    scores = {
        route: sum(term in question_lower for term in terms)
        for route, terms in keywords.items()
    }
    scores["math"] += sum(
        symbol in question for symbol in ["+", "-", "*", "/", "=", "^", "√", "π", "x", "y"]
    )
    for route in ("memory", "code", "web"):
        if scores[route] > 0:
            return route
    if scores["hash"] > scores["math"] and scores["hash"] > 0:
        return "hash"
    if scores["math"] > 0:
        return "math"
    return "unknown"


def extract_arguments(question: str, tool: str) -> dict[str, object] | None:
    """Extract deterministic arguments for routes that expose structured inputs."""
    normalized = question.lower()
    if tool == "memory":
        pair = re.search(r"(\d+)\s+(?:and|with)\s+(\d+)", normalized)
        if pair and re.search(r"\b(remember|store|save)\b", normalized):
            return {
                "action": "store",
                "key": pair.group(1),
                "value": pair.group(2),
            }
        number = re.search(r"\d+", normalized)
        if number:
            return {"action": "retrieve", "key": number.group(0)}
    elif tool == "web":
        url = re.search(r"https?://[^\s]+", question)
        if url:
            return {"url": url.group(0).rstrip(".,;:!?)]}")}
    elif tool == "hash":
        algorithms = re.findall(r"\b(md5|sha1|sha224|sha256|sha384|sha512)\b", normalized)
        if algorithms:
            return {"algorithms": algorithms}
    return None
