# Building AI Infra

This is the Hugo project that powers my *Building AI Infra* blog. It uses a small custom theme (`themes/minimal`) that focuses on clean typography, responsive layouts, and a dark-mode toggle.

## Local development

```bash
hugo server --buildDrafts --disableFastRender
```

The server becomes available at `http://localhost:1313` with hot reload enabled.

## Adding new posts

Use the default archetype to pre-fill front matter:

```bash
hugo new posts/my-new-note.md
```

Feel free to update the fields it generates (`description`, `excerpt`, `tags`, `categories`) before committing.*** End Patch
