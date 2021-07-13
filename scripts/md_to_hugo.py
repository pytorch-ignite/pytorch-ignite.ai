"""Script to prepare plain markdown files for the Hugo.

Example Usage:
    to prepare all the md files in a content/docs/tutorials/
    run this command from the repo root dir:
    python ./scripts/md_to_hugo.py ./content/docs/tutorials/
"""
import glob
import os
import shutil
import sys

PATTERN = "../../../static/images/notebooks/"
IMG_DIR = "/images/notebooks/"


def main(dir):
    if dir is None:
        raise ValueError(f"Please specify directory containing md files")
    # Iterate over md files in a dir
    for md in glob.iglob(f"{dir}/*md"):
        # Ignore _index.md, etc
        if not os.path.basename(md).startswith("_"):
            # Prepare md for hugo acceptable format
            with open(md, "r") as f1, open(f"{md}.fixed", "w") as f2:
                for i, line in enumerate(f1):
                    if PATTERN in line:
                        line = line.replace(PATTERN, IMG_DIR)
                    # Get title (for now its the first line)
                    if i == 0:
                        title = line.strip()
                        hugo_meta = f"---\ntitle: {title}\ninclude_footer: true\n---\n"
                        f2.write(hugo_meta)
                    f2.write(line)

            # Replace md file with correct one
            shutil.move(f"{md}.fixed", md)


if __name__ == "__main__":
    dir = sys.argv[1] if len(sys.argv) == 2 else None
    main(dir)
