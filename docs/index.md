# Distill-Align Documentation

Welcome to the Distill-Align documentation!

## Table of Contents

- [Getting Started](getting-started.md) — Installation and quick start
- [Configuration](configuration.md) — Config file and environment variables
- [CLI Reference](cli-reference.md) — All CLI commands and options

## Overview

Distill-Align is a CLI/Python framework that automates the generation of high-quality, fine-tuning datasets from raw domain data (PDFs, Markdown, Codebases). It uses frontier reasoning models as teachers to transform raw content into structured instruction-following formats optimized for Unsloth Studio fine-tuning.

## Core Components

### Ingestion Module

Supports multiple file formats with semantic-aware chunking:
- **Markdown/PDF/Code** — Smart chunking by headers, definitions
- **DOCX/HTML** — Office documents and web pages
- **Jupyter Notebooks** — Code + markdown + outputs
- **JSON/CSV** — Structured data

### Synthesis Module

Generates structured conversations using:
- **Socratic Transformer** — Conversational multi-turn Q&A
- **Scaffold Action** — Strips filler, extracts clean output
- **Multiple modes** — teach, debug, review, qa, explain
- **External prompts** — Customizable `.j2` templates
- **Caching & checkpoints** — Resume from failures

### Export Module

Outputs to multiple formats:
- **ShareGPT** — Standard multi-turn format
- **Alpaca** — Instruction-following format
- **ChatML** — Qwen/OpenHermes compatible
- **Conversation** — Generic format
- **Auto-splits** — Train/val/test with stratification
- **Dataset cards** — HuggingFace-style README

### CLI & TUI

Rich command-line interface with:
- **Subcommands** — ingest, synthesize, export, validate, jobs, config
- **Live progress** — Rich tables and progress bars
- **Cost estimation** — Token counting for OpenAI models
- **Job management** — List, resume, delete synthesis jobs
- **TUI dashboard** — Real-time monitoring with Textual
