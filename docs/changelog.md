# Changelog

All notable changes to Distill-Align are documented here.

See also: [CHANGELOG.md](https://github.com/omargargoum/Distill-Align/blob/main/CHANGELOG.md) on GitHub.

## [0.1.1] - 2026-06-18

### Fixed

- Textual `_auto_refresh` attribute conflict
- MyPy type errors across multiple modules
- Ruff SIM108 ternary formatting
- CI: ruff, mypy, mkdocs warnings, benchmark `--no-cov`, TUI skip in non-TTY
- Bandit skips, install groups, docs, TUI test

### Added

- Cost tracking and streaming JSONL/Parquet export
- CLI `cost-report` command for usage estimation
- `--max-tokens` option for LLM call token limits
- Arabic README translation
- Comprehensive documentation (MkDocs)

## [0.1.0] - 2026-06-16

### Added

- **Ingestion pipeline** with 9 file loaders (Markdown, PDF, DOCX, HTML, Jupyter, JSON, CSV, Code, Text)
- **Synthesis pipeline** with 6 LLM providers (OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM)
- **Socratic Transformer** — converts raw content into multi-turn Q&A conversations
- **Scaffold Action** — cleans and extracts structured content from assistant responses
- **LLM-as-Judge evaluation** — automated quality scoring on 5 criteria (relevance, coherence, correctness, completeness, safety)
- **DPO preference pair generation** for Direct Preference Optimization training
- **Export pipeline** with 7 output formats (ShareGPT, Alpaca, ChatML, HuggingFace, JSONL, Parquet, Conversation)
- **Streaming export** for large datasets without full memory load
- **Dataset validation** — structural checks, quality scoring, deduplication, and statistics
- **PII filtering** — detects and redacts secrets and personal information
- **Job management** with checkpoint/resume for crash recovery
- **TUI dashboard** with real-time monitoring via Textual
- **Unsloth integration** — auto-generated training scripts
- **Security scanning** — Bandit, Safety, pip-audit, CodeQL, dependency review
- **CI/CD** — lint, type-check, test matrix (3 Python versions), security scan, docs deploy
