# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hugo-based static site (PaperMod theme) for a technical blog about AI infrastructure. Deployed to GitHub Pages at https://nvbkdw.github.io/building-ai-infra.

## Commands

```bash
# Local development server with live reload
hugo server

# Production build (mirrors CI)
hugo --gc --minify --baseURL "https://nvbkdw.github.io/building-ai-infra"
```

There are no test commands. Always verify changes build cleanly with `hugo server` before committing.

## Deployment

Push to `main` triggers GitHub Actions (`.github/workflows/hugo.yaml`) which builds with Hugo v0.152.2 and deploys to GitHub Pages. No manual deploy steps.

## Architecture

- **Content**: All blog posts live in `content/blogs/` as Markdown files with YAML frontmatter
- **Layouts**: Custom templates in `layouts/` override the PaperMod theme in `themes/PaperMod/`
  - `layouts/blogs/single.html` — custom blog post layout
  - `layouts/partials/` — reusable components (header, footer, math, TOC, etc.)
- **Assets**: CSS in `assets/css/` (core theme vars + component-specific stylesheets)
- **Static files**: Images, PDFs, favicons in `static/`
- **Config**: `config.yml` is the single Hugo configuration file

The homepage uses PaperMod's profile mode (not archive mode). Math rendering uses MathJax. Syntax highlighting uses the "autumn" style.

## Blog Post Conventions

Every blog post requires this frontmatter:

```yaml
---
title: "Post Title"
date: YYYY-MM-DD
tags: ["tag1", "tag2"]
author: "Ryan H."
description: "Brief description"
summary: "Brief summary"
cover:
    image: "filename.png"
    alt: "Alt text"
    relative: true
---
```

- File naming: kebab-case (e.g., `gpu-kernel-programming-101.md`)
- Headings: start at H2 (`##`); H1 is reserved for the page title
- Math: inline `$...$`, display `$$...$$` (MathJax enabled)
- Code blocks: fenced with language specifier; copy buttons enabled
- Tags: lowercase, hyphens for multi-word (e.g., `distributed-training`, `gpu-kernel`)
- Images: use relative paths, place cover images in `static/`
