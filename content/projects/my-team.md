---
title: "My Team"
description: "A Kanban board where every card is an AI-powered development environment. Drag a task to 'In Progress' and a dedicated Claude Code session spins up — with its own Git branch, file watcher, and streaming AI chat — turning your task tracker into a hands-on coding workbench."
summary: "Kanban board with AI-powered development environments per card"
tags: ["typescript", "ai-agents", "developer-tools"]
externalUrl: "https://github.com/nvbkdw/my-team"
weight: 2
hidemeta: true
---

A Kanban board where every card is an AI-powered development environment. Drag a task to "In Progress" and a dedicated Claude Code session spins up — with its own Git branch, file watcher, and streaming AI chat — turning your task tracker into a hands-on coding workbench.

## Why This Exists

Traditional project management tools live in one tab; your IDE lives in another. Context switching between the two is constant. **My Team** collapses that gap: each card on the board is backed by a real Git worktree and a persistent Claude Code session that can read, write, and reason about the code. You plan work _and_ execute it in the same interface.

## Key Features

- **Kanban board** with drag-and-drop (backlog / priority / in progress / done)
- **One Claude worker per card** — isolated child processes with automatic restart and crash recovery (up to 5 concurrent)
- **Git worktrees** — each card gets its own working directory, so multiple tasks work on the same repo without conflicts
- **Real-time streaming** — Claude's responses stream token-by-token over WebSocket directly into the card's chat panel
- **Built-in diff viewer** — split or unified diffs powered by `@git-diff-view/react`, right next to the conversation
- **GitHub PR integration** — create, view, and review pull requests without leaving the board
- **File browser & editor** — browse the card's branch directory and read/write files from the UI

## Architecture

| Layer  | Stack                                                |
| ------ | ---------------------------------------------------- |
| Client | React 19, Vite 6, Zustand 5, Tailwind CSS 3, dnd-kit |
| Server | Express 5, better-sqlite3, ws, simple-git            |
| AI     | Claude Agent SDK (@anthropic-ai/claude-agent-sdk)    |
| VCS    | Git worktrees, Octokit (GitHub API)                  |

**Tech stack**: TypeScript, React, Express, Claude Agent SDK, SQLite, Git worktrees

[View on GitHub](https://github.com/nvbkdw/my-team)
