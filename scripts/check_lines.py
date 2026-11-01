"""Enforce the repository's per-module 350-line readability guideline."""

from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    files = [file for folder in ("src", "tests", "evaluation", "scripts")
             for file in (project / folder).rglob("*.py")]
    oversized = [(str(file.relative_to(project)), len(file.read_text(encoding="utf-8").splitlines()))
                 for file in files if len(file.read_text(encoding="utf-8").splitlines()) > 350]
    for name, count in oversized:
        print(f"Too long: {name} ({count} lines; maximum 350)")
    if not oversized:
        print(f"OK: {len(files)} Python files are all at or below 350 lines")
    return 1 if oversized else 0


if __name__ == "__main__":
    raise SystemExit(main())
