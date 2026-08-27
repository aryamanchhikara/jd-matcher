"""Load job descriptions from data/jds/*.txt.

Lines beginning with # at the top of the file (before the first
non-comment line) are treated as loader comments and stripped, so
placeholder files can carry human instructions without polluting the
embedded text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JD_DIR = ROOT / "data" / "jds"


@dataclass
class JobDescription:
    slug: str      # filename without extension, used as ID everywhere
    title: str     # first non-comment line
    text: str      # full text to embed (including title)


def _strip_leading_comments(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_leading = True
    for line in lines:
        if in_leading and line.startswith("#"):
            continue
        in_leading = False
        out.append(line)
    return "\n".join(out).strip()


def load_all() -> list[JobDescription]:
    files = sorted(JD_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No JDs found in {JD_DIR}")
    jds: list[JobDescription] = []
    for f in files:
        text = _strip_leading_comments(f.read_text(encoding="utf-8"))
        if not text:
            # Placeholder file: skip with a warning-in-title so the
            # heatmap still has a row and the human notices.
            jds.append(JobDescription(
                slug=f.stem,
                title=f"[EMPTY PLACEHOLDER: {f.stem}]",
                text=f"[Empty placeholder file: {f.stem}]",
            ))
            continue
        title = text.splitlines()[0].strip()
        jds.append(JobDescription(slug=f.stem, title=title, text=text))
    return jds
