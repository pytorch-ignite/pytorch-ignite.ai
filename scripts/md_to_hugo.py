"""Script to prepare plain markdown files for the Hugo.

Example Usage:
    to prepare all the md files in a content/docs/tutorials/
    run this command from the repo root dir:
    python ./scripts/md_to_hugo.py ./content/docs/tutorials/
"""
import re
import sys
from pathlib import Path


def main(dir: str):
    if dir is None:
        raise ValueError("Please specify directory containing md files")

    path = Path(dir)

    for file in path.rglob("*.md"):
        if not file.name.startswith("_"):
            text = (
                file.read_text("utf-8")
                .replace("<!-- ", "", 1)
                .replace(" -->", "", 1)
                .replace(str(path.cwd() / 'static/_images'), '/_images')
            )
            file.write_text(text, "utf-8")


if __name__ == "__main__":
    dir = sys.argv[1] if len(sys.argv) == 2 else None
    main(dir)
