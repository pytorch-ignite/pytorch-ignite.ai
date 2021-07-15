# pytorch-ignite.ai

The website is made with [Gohugo](https://gohugo.io). You will need to install extended version of Hugo 0.83.1.

## Available usages

- Use `hugo new posts/file-name.md` to make a new markdown file inside `content/posts`. That file will include frontmatter defined in `archetypes/posts.md` and will be inside `content/posts`.

> Tutorials, Guides, ... can also be created inside `archetypes` directory.

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

It also accpet optional title. For eg,

For custom note title:

```sh
{{% tip "Engine's device NOTE" %}}
Some text
{{% /tip %}}
```
