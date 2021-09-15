# pytorch-ignite.ai

This website is built with [Hugo](https://gohugo.io). You will need to install the extended version of Hugo 0.88.1.

> Blog posts relating to PyTorch-Ignite are welcomed for pull requests.

## Development

Current directory structure:

```sh
.
├── src                 # the main markdown content
│   ├── about           # About PyTorch-Ignite
│   ├── blog            # blog posts
│   ├── guides          # PyTorch-Ignite How-to guides
│   └── tutorials       # PyTorch-Ignite tutorials
├── scripts             # various CI/CD scripts
├── static
│   ├── examples        # https://github.com/pytorch-ignite/examples submodule
│   └── _images         # static images
│   └── blog            # Blog posts in jupyter notebook format
└── themes/ignite/assets/_hugo
│   ├── css             # css files (WindiCSS generated files + hand written ones)
│   ├── js              # js files
└── themes/ignite/layouts
│   ├── _default        # default layout theme
│   ├── partials        # partial layouts (can be reused)
│   ├── shortcodes      # Hugo shortcodes (tip, info, warning, danger & details)
│   ├── taxonomy        # layouts for tags
```

After installing Hugo, start a dev server with

```sh
hugo server  # dev server at localhost:1313
```

If the included submodules are out of sync, run

```sh
git submodule update --remote -- static/examples
```

Hugo and theme level configurations are defined in [config.yaml](./config.yaml).

## Available usages

- Create markdown files with necessry frontmatter. You can reference from the exisiting files.
If custom slug url is desired, `slug` in frontmatter can be used.

- New blog posts automatically get previous and next blog post links.

- When creating new templates, take a look at `layouts/_default` and use available template parts inside `layouts/partials`.

- We can also use [Hugo shortcodes](https://gohugo.io/content-management/shortcodes/#readout) for `INFO`, `TIP`, `WARNING`, `DANGER`, and `DETAILS` admonitions.
  Use it like `{{<>}}` and `{{</>}}`.
  ```sh
  {{<highlight go>}} A bunch of code here {{</highlight>}}
  ```

For info:

```sh
{{<info>}}
Some text
{{</info>}}
```

For tip:

```sh
{{<tip>}}
Some text
{{</tip>}}
```

For warning:

```sh
{{<warning>}}
Some text
{{</warning>}}
```

For danger:

```sh
{{<danger>}}
Some text
{{</danger>}}
```

For details:

```sh
{{<details>}}
Some text
{{</details>}}
```

It also accept optional title. For example,

For custom note title:

```sh
{{<tip "Engine's device NOTE">}}
Some text
{{</tip>}}
```

## Writing contents

Contents are usually written in markdown files in the `blog` directory or in separate markdown files like [`ecosystem.md`](./src/ecosystem.md).

To write How-to-guides or tutorials, make a pull request to [examples repo](https://github.com/pytorch-ignite/examples). GitHub bot will update the files in this repo every 6 hours.

To write a blog post in jupyter notebook format, use [`generate.py`](https://github.com/pytorch-ignite/examples/blob/main/generate.py) script in `static/examples` to generate a notebook with pre-defined frontmatters.

**Blog post filenames must be like `year-month-day-title.{md,ipynb}`. For examples, see [`2020-09-10-pytorch-ignite.md`](./src/blog/2020-09-10-pytorch-ignite.md).**
