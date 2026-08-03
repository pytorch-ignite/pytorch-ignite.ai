import json
import os
import shutil
import subprocess
from datetime import datetime

import pytest

today = datetime.now().strftime("%Y-%m-%d")

new_notebook_empty = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
new_notebook_empty = json.dumps(new_notebook_empty, indent=4)

cell_front_matter = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "<!-- ---\n",
        "title: <required-title>\n",
        f"date: {today}\n",
        "downloads: true\n",
        "weight: <required-weight> See: https://github.com/pytorch-ignite/examples/issues/30\n",
        "summary: <use either this or the `<!--more-->` tag below to provide summary for this notebook, "
        "and delete the other>\n"
        "tags:\n",
        "  - <required-tag>\n",
        "--- -->\n",
        "\n",
        "# title-placeholder\n",
        "\n",
        "<If you are not using the `summary` variable above, use this space to "
        "provide a summary for this notebook.>\n",
        "<Otherwise, delete the `<!--more-->` below.>",
        "\n",
        "<!--more-->",
    ],
}


@pytest.mark.parametrize("name", ["dummy_notebook", "dummy_notebook.ipynb"])
def test_new_notebook_creation(name, tmp_path):
    notebook_path = os.path.join(tmp_path, name)

    output = subprocess.check_output(["python", "generate.py", notebook_path]).decode("utf-8")

    if not notebook_path.endswith(".ipynb"):
        notebook_path = notebook_path + ".ipynb"

    assert output == f"Generated {notebook_path}\n"


def test_existing_blank_notebook(tmp_path):
    notebook_path = os.path.join(tmp_path, "dummy_notebook_empty.ipynb")
    with open(notebook_path, "w") as f:
        f.write(new_notebook_empty)

    output = subprocess.check_output(["python", "generate.py", notebook_path]).decode("utf-8")

    assert output == f"Added frontmatter to {notebook_path}\n"


def test_existing_non_empty_notebook(tmp_path):
    notebook_name = "01-getting-started.ipynb"
    notebook_path = os.path.join(tmp_path, notebook_name)
    shutil.copyfile(os.path.join("./tutorials/beginner", notebook_name), notebook_path)

    output = subprocess.check_output(["python", "generate.py", notebook_path]).decode("utf-8")

    assert output == f"Added frontmatter to {notebook_path}\n"

    # Check to make sure its added as the first cell
    with open(notebook_path) as fp:
        content = json.load(fp)
    assert content["cells"][0] == cell_front_matter


@pytest.mark.parametrize("name", ["dummy_notebook_empty", "dummy_notebook_empty.ipynb"])
def test_front_matter_multiple_times(name, tmp_path):
    notebook_path = os.path.join(tmp_path, name)

    # This will create a notebook with frontmatter
    _ = subprocess.check_output(["python", "generate.py", notebook_path])

    # Second call should not add frontmatter again
    output = subprocess.check_output(["python", "generate.py", notebook_path]).decode("utf-8")

    if not notebook_path.endswith(".ipynb"):
        notebook_path = notebook_path + ".ipynb"

    assert output == f"Frontmatter cell already exists in {notebook_path}. Exiting\n"
    # Check to make sure only added once.
    with open(notebook_path) as fp:
        content = json.load(fp)

    if len(content["cells"]) > 1:
        assert content["cells"][0] != content["cells"][1]
