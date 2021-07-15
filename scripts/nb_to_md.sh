#!/bin/bash
# Usage: nb_to_md.sh <notebook_dir> <target_dir>
# Example: ./scripts/nb_to_md.sh examples ./content/docs/tutorials
# Convert from ipynb to plain md
jupyter nbconvert --to markdown $1/*.ipynb --output-dir $2 --NbConvertApp.output_files_dir=../../../static/images/notebooks
# Make Hugo compatible md
python scripts/md_to_hugo.py $2
