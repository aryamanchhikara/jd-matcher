"""Structured constraint parsing and filtering.

The parser is deliberately simple: regex + keyword matching. That is
the whole point of the exercise. Constraints like "5+ years" and
"onsite in New York" are the kind of thing that embeddings blur but a
one-line rule handles crisply.

The parse is a best-effort read of the free-text query; the notebook
also exposes each field as an override so the demo can be steered by
hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable


# Cities we recognise. Extend as needed. Keeps the demo honest:
# there is no clever NER here, just a lookup.
_CITY_ALIASES = {
    "new york": "New York",
    "new york city": "New York",
    "nyc": "New York",
    "san francisco": "San Francisco",
    "sf": "San Francisco",
    "the bay area": "San Francisco",
    "seattle": "Seattle",
    "berlin": "Berlin",
    "london": "London",
    "remote": None,   # location=remote handled via work_arrangement, not city
}

_ARRANGEMENT_KEYWORDS = {
    "onsite":  ["onsite", "on-site", "in office", "in the office", "in person"],
    "hybrid":  ["hybrid"],
    "remote":  ["remote", "fully remote", "work from home", "wfh"],
}

_SENIORITY_KEYWORDS = {
    "senior":  ["senior", "sr.", "staff", "principal", "lead"],
    "mid":     ["mid", "mid-level", "intermediate"],
    "junior":  ["junior", "jr.", "entry", "new grad", "graduate"],
}


@dataclass
class Constraints:
    """Parsed structured constraints from a free-text query.

    None means "no constraint on this field". The keyword hits are
    preserved so the UI can show what was picked up and why.
    """
    location: str | None = None
    work_arrangement: str | None = None   # 'onsite' | 'hybrid' | 'remote'
    min_years: int | None = None
    seniority: str | None = None          # 'senior' | 'mid' | 'junior'
    skills: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def with_overrides(self, **kwargs) -> "Constraints":
        """Return a copy with any non-None kwargs applied."""
        overrides = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **overrides)


# A short skills vocabulary is enough for the demo. We keep it explicit
# so the notebook reader can see what is and isn't recognised.
_SKILL_VOCAB = [
    "go", "golang", "python", "rust", "java", "scala", "typescript",
    "javascript", "react", "ruby", "c++", "kubernetes", "postgres",
    "sql", "spark", "pytorch", "jax", "distributed systems",
]


def parse_query(q: str) -> Constraints:
    """Extract structured constraints from a free-text query.

    Deterministic, cheap, easy to reason about. Not clever.
    """
    ql = q.lower()
    trace: list[str] = []
    c = Constraints()

    # Location: longest alias first so "new york city" beats "new york".
    for alias in sorted(_CITY_ALIASES.keys(), key=len, reverse=True):
        if alias in ql and _CITY_ALIASES[alias] is not None:
            c.location = _CITY_ALIASES[alias]
            trace.append(f"location={c.location} (matched '{alias}')")
            break

    # Work arrangement.
    for arrangement, keywords in _ARRANGEMENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", ql):
                c.work_arrangement = arrangement
                trace.append(f"work_arrangement={arrangement} (matched '{kw}')")
                break
        if c.work_arrangement:
            break

    # Years of experience: match "5+ years", "5 years", "at least 5 years".
    m = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?)", ql)
    if m:
        c.min_years = int(m.group(1))
        trace.append(f"min_years={c.min_years} (matched '{m.group(0)}')")

    # Seniority.
    for level, keywords in _SENIORITY_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", ql):
                c.seniority = level
                trace.append(f"seniority={level} (matched '{kw}')")
                break
        if c.seniority:
            break

    # Skills.
    for skill in _SKILL_VOCAB:
        if re.search(rf"\b{re.escape(skill)}\b", ql):
            c.skills.append(skill)
    if c.skills:
        trace.append(f"skills={c.skills}")

    c.trace = trace
    return c


def _profile_years(p: dict) -> int:
    y = p.get("years_of_experience")
    return int(y) if y is not None else 0


def passes(profile: dict, c: Constraints) -> tuple[bool, list[str]]:
    """Return (passes, reasons_for_rejection).

    Reasons are collected so the UI can explain why a semantically
    strong candidate was filtered out. That explanation is the whole
    point of the exercise.
    """
    reasons: list[str] = []

    if c.location:
        prof_loc = (profile.get("location") or "").strip()
        if not prof_loc:
            reasons.append(f"missing location; JD requires {c.location}")
        elif prof_loc != c.location:
            reasons.append(f"location {prof_loc!r} != required {c.location!r}")

    if c.work_arrangement:
        prof_arr = (profile.get("work_arrangement") or "").strip().lower()
        # A candidate open to X or more restrictive is fine only where
        # it clearly matches. We are strict on purpose: the point of the
        # demo is that hard constraints are hard.
        if prof_arr != c.work_arrangement:
            reasons.append(
                f"work arrangement {prof_arr!r} != required {c.work_arrangement!r}"
            )

    if c.min_years is not None:
        yrs = _profile_years(profile)
        if yrs < c.min_years:
            reasons.append(f"has {yrs} yrs, requires {c.min_years}+")

    # Seniority is soft: cosine is often good enough for the "senior"
    # vs "junior" gestalt. We skip enforcing it in the filter and let
    # it come through in the ranking. (Trivial to add if you want it.)

    return (len(reasons) == 0), reasons


def apply(profiles: Iterable[dict], c: Constraints) -> tuple[list[dict], list[dict]]:
    """Split profiles into (passed, rejected_with_reasons)."""
    kept: list[dict] = []
    dropped: list[dict] = []
    for p in profiles:
        ok, reasons = passes(p, c)
        if ok:
            kept.append(p)
        else:
            dropped.append({**p, "_reject_reasons": reasons})
    return kept, dropped
