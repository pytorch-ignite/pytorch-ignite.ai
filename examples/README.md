# PyTorch-Ignite Examples, tutorials and how-to guides

Notebooks in this repo requires the essential frontmatters to be used
when rendering on the [website](https://pytorch-ignite.ai).

To contribute the notebooks, please use [`generate.py`](./generate.py)
script to generate the notebook.

**Usage:**

```sh
# python generate.py <notebook-names>...
python generate.py yolo
# > Generated /workspace/yolo.ipynb
```

Alternatively, you can run `generate.py` on your existing notebooks to add the required frontmatter cell to them.
```sh
# If your completed tutorial is present in /workspace/yolo.ipynb
python generate.py yolo
# > Added frontmatter to /workspace/yolo.ipynb
```
This will add the necessary frontmatter cell at the top of the notebook, now you need to open it and update the values.

See more in [`generate.py`](./generate.py).
