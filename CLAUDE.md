# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hugo-based static site (PaperMod theme) for a technical blog about AI infrastructure. Deployed to GitHub Pages at the custom domain https://www.ryhuang.blog (`static/CNAME`).

## Commands

```bash
# Local development server with live reload
hugo server

# Production build (mirrors CI)
hugo --gc --minify --baseURL "https://www.ryhuang.blog/"

# New post with the standard frontmatter
hugo new blogs/my-post-title.md
```

There are no test commands. Always verify changes build cleanly with `hugo server` before committing. For visual changes, open the site in a browser (or take screenshots) in **both** color modes — the header toggle switches them.

## Deployment

Push to `main` triggers GitHub Actions (`.github/workflows/hugo.yaml`) which builds with Hugo v0.152.2 and deploys to GitHub Pages. No manual deploy steps.

## Architecture

The design follows [lilianweng.github.io](https://lilianweng.github.io/) (Lil'Log): a single 720px reading column, header aligned to that column, a welcome card followed by bordered post cards on a tinted background, a labelled `Date | Estimated Reading Time | Author` meta line, and a collapsible table of contents. The palette is a warm "hearth" theme — dark firewood browns with an ember accent by default, a parchment light mode behind the toggle.

- **Theme**: PaperMod is vendored in `themes/PaperMod/` (checked in, not a submodule). **Do not edit theme files.** All customization lives in the project, layered on top of the theme through PaperMod's extension points:
  - `assets/css/extended/*.css` — concatenated after the theme's CSS, so plain redefinitions win. Files are numbered to control order: `00-palette.css` (all color/size variables for light + dark), `10-layout.css` (type, header, welcome card, post cards, footer, archive, tags), `20-post.css` (single-post typography, figures, tables, code, TOC, post footer), `30-projects.css` (project card grid), `40-media.css` (responsive tweaks).
  - `assets/css/includes/chroma-styles.css` — syntax-highlighting classes (gruvbox base). Regenerate with `hugo gen chromastyles --style=<name>`; requires `markup.highlight.noClasses: false` in `config.yml`.
  - `layouts/partials/extend_head.html` — hooks KaTeX (`layouts/partials/math.html`) into `<head>` when `math` is enabled.
  - `layouts/partials/home_info.html` — homepage welcome card (avatar + greeting + intro + social icons); content comes from `params.homeInfoParams` in `config.yml`.
  - `layouts/partials/post_meta.html` — the labelled meta line used on cards, archive entries, and post headers.
  - `layouts/projects/list.html` — card grid for the Projects section (reads `description`, `tags`, `externalUrl`, `cover` from each project's frontmatter).
  - Markdown render hooks: `layouts/_default/_markup/render-link.html` (external links open in a new tab), `render-passthrough-{block,inline}.html` (math delimiters passed through to KaTeX), `layouts/blogs/_markup/render-image.html` (rewrites `/static/...` image paths).
- **Content**: blog posts in `content/blogs/`; standalone pages `content/about.md`, `content/archives.md` (PaperMod `archives` layout); project pages in `content/projects/`.
- **Static files**: images, PDFs, favicons, `CNAME` in `static/`. Post images are referenced as `/static/<file>` (the render hook fixes the path).
- **Config**: `config.yml` is the single Hugo configuration file. Homepage = PaperMod home-info mode (welcome card + post list); navigation = `menu.main`; only `blogs` is a main section (home list + archive); taxonomies are disabled (`tags:` frontmatter is metadata only, no `/tags/` pages); TOC is enabled for `/blogs/**` only via `cascade`.

Math rendering uses KaTeX (auto-render). Syntax highlighting is class-based Chroma with the gruvbox palette.

## Blog Post Conventions

Every blog post requires this frontmatter (see `archetypes/blogs.md`):

```yaml
---
title: "Post Title"
date: YYYY-MM-DD
tags: ["tag1", "tag2"]
author: "Ryan H."
description: "Brief description"   # shown under the title and used for SEO
summary: "Brief summary"           # shown on the post card
cover:                             # optional
    image: "filename.png"
    alt: "Alt text"
    relative: true
---
```

- File naming: kebab-case (e.g., `gpu-kernel-programming-101.md`)
- Headings: start at H2 (`##`); H1 is reserved for the page title. The TOC covers levels 1–3 so older posts that use `#` still render correctly.
- Math: inline `$...$`, display `$$...$$` (KaTeX)
- Code blocks: fenced with language specifier; copy buttons enabled
- Figures: use `{{< figure src="..." caption="..." >}}` for auto-numbered "Figure N." captions; plain markdown images render centered without a caption
- Tags: lowercase, hyphens for multi-word (e.g., `distributed-training`, `gpu-kernel`); used as SEO keywords only
- Images: place in `static/` and reference as `/static/<file>`
