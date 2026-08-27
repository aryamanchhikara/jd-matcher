"""Load candidate profiles.

Each profile is a markdown file with YAML frontmatter. The frontmatter
holds the STRUCTURED FIELDS (location, seniority, years_of_experience,
work_arrangement, is_fabricated, source_url). The prose below is what
gets embedded.

For real karrouhq profiles, the fetcher pulls .md from
karrouhq.com/{username}.md and caches it. A curator then annotates the
structured fields by hand, because parsing them out of prose is exactly
the failure mode this whole prototype is arguing against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "data" / "profiles"
CACHE_DIR = ROOT / "data" / "cache" / "profiles"


@dataclass
class Profile:
    username: str                 # short id used in tables and file names
    display_name: str
    location: str | None
    work_arrangement: str | None  # 'onsite' | 'hybrid' | 'remote' | None
    years_of_experience: int | None
    seniority: str | None         # 'junior' | 'mid' | 'senior'
    prose: str                    # what actually gets embedded
    is_fabricated: bool
    source_url: str | None

    def as_filter_dict(self) -> dict:
        return {
            "username": self.username,
            "location": self.location,
            "work_arrangement": self.work_arrangement,
            "years_of_experience": self.years_of_experience,
            "seniority": self.seniority,
        }


def fetch_karrouhq(username: str, *, force: bool = False) -> str | None:
    """Download karrouhq.com/{username}.md, cache to disk. Returns text or None."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{username}.md"
    if cache_file.exists() and not force:
        return cache_file.read_text(encoding="utf-8")

    url = f"https://karrouhq.com/{username}.md"
    r = requests.get(url, headers={"User-Agent": "jd-matcher/0.1"}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    cache_file.write_text(r.text, encoding="utf-8")
    return r.text


def load_profile(path: Path) -> Profile:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path} is missing YAML frontmatter")
    _, fm, body = raw.split("---\n", 2)
    meta = yaml.safe_load(fm)
    return Profile(
        username=meta["username"],
        display_name=meta.get("display_name", meta["username"]),
        location=meta.get("location"),
        work_arrangement=meta.get("work_arrangement"),
        years_of_experience=meta.get("years_of_experience"),
        seniority=meta.get("seniority"),
        prose=body.strip(),
        is_fabricated=bool(meta.get("is_fabricated", False)),
        source_url=meta.get("source_url"),
    )


def load_all() -> list[Profile]:
    """Load every *.md in data/profiles/ in filename order."""
    files = sorted(PROFILES_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No profiles found in {PROFILES_DIR}. "
            "Populate the directory before running the notebook."
        )
    return [load_profile(f) for f in files]
