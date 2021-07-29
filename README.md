# pytorch-ignite.ai

This website is built with [Hugo](https://gohugo.io). You will need to install the extended version of Hugo 0.83.1.

> Blog posts relating to PyTorch-Ignite are welcomed for pull requests.

## Development

Current directory structure:

```sh
.
├── archetypes      # frontmatter template for markdown files
├── assets          # website assets (css + js)
│   ├── css
│   └── js
├── content         # the main markdown content
│   ├── guide       # PyTorch-Ignite How-to guides
│   ├── posts       # blog posts
│   └── tutorials   # PyTorch-Ignite tutorials
├── layouts
│   ├── _default    # default layout theme
│   ├── partials    # partial layouts (can be reused)
│   ├── posts       # layouts for blog posts
│   ├── shortcodes  # Hugo shortcodes
│   ├── taxonomy    # layouts for tags
│   └── tutorials   # layouts for tutorials
├── scripts         # various CI/CD scripts
├── static
│   ├── examples    # https://github.com/pytorch-ignite/examples submodule
│   └── images      # static images
└── themes
    └── hugo-fresh
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

- Use `hugo new posts/file-name.md` to make a new markdown file inside `content/posts`. That file will include frontmatter defined in `archetypes/posts.md`. If custom slug url is desired, `slug` in frontmatter can be used.

- New blog posts automatically get previous and next blog post links.

- When creating new templates, try to create a one level directory (currently no nested) inside `layouts` with the directory name that will be used inside `content`. For example, `content/posts` has its respective templates inside `layouts/posts`.

- When creating new templates, take a look at `layouts/_default`, `layouts/posts`, and use available template parts inside `layouts/partials`.

- We can also use [Hugo shortcodes](https://gohugo.io/content-management/shortcodes/#readout) for `TIP`, `WARNING`, `DANGER`, and `DETAILS` admonitions.
  There is 2 syntaxes for shortcodes (`{{%%}}` and `{{<>}}`).
  ```sh
  {{% mdshortcode %}}Stuff to `process` in the *center*.{{% /mdshortcode %}}
  ```
  ```sh
  {{< highlight go >}} A bunch of code here {{< /highlight >}}
  ```

For tip:

```sh
{{% tip %}}
Some text
{{% /tip %}}
```

For warning:

```sh
{{% warning %}}
Some text
{{% /warning %}}
```

For danger:

```sh
{{% danger %}}
Some text
{{% /danger %}}
```

For details:

```sh
{{% details %}}
Some text
{{% /details %}}
```

It also accept optional title. For example,

For custom note title:

```sh
{{% tip "Engine's device NOTE" %}}
Some text
{{% /tip %}}
```
