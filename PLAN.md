# Distill-Align: Production Hardening & Feature Complete Plan

- [x] **Phase 1 — Security & Correctness**
  - [x] 1. Fix SSRF (C2): URL validation, private-IP blocking, redirect cap, body size cap
  - [x] 2. Fix `--no-cache` (C1): thread `use_cache` through synthesize → worker
  - [x] 3. Bound input sizes (C3): max-bytes guard on json.load — json_utils.py, cache.py, all callers
  - [x] 4. Harden Jinja2 template (H1): json.dumps()-escape strings via `| tojson` filter
  - [x] 5. Tighten Bandit (H3): remove B101 skip, convert asserts to exceptions
  - [x] 6. PII + secret scan stage (H4): add pii_filter.py with PII + secret patterns, integrated into ingestion pipeline

- [ ] **Phase 2 — Quality Gates & CI**
  - [ ] 7. Coverage gate: `--cov-fail-under=70`
  - [ ] 8. HTTP mocking: add pytest-httpx, write client tests
  - [ ] 9. Worker/pipeline tests: concurrency, retry, cache, resume
  - [ ] 10. Installed-package smoke test in CI
  - [ ] 11. Bump tooling: Ruff, pre-commit, drop click pin
  - [ ] 12. SQLite thread-safety review
  - [ ] 13. Supply chain: pip-audit + dependency-review-action

- [ ] **Phase 3 — Modern Feature Parity**
  - [ ] 14. Add HF `messages` formatter
  - [ ] 15. Wire judge into synthesis pipeline
  - [ ] 16. DPO/preference generation
  - [ ] 17. Structured outputs (response_format: json_object)
  - [ ] 18. More providers: Anthropic, Gemini, Azure-AD
  - [ ] 19. Streaming JSONL + Parquet export
  - [ ] 20. Cost tracking

- [ ] **Phase 4 — Polish**
  - [ ] 21. Fix code smells; tighten mypy
  - [ ] 22. Beef up SECURITY.md + .well-known/security.txt
  - [ ] 23. Wire mkdocstrings for API docs
  - [ ] 24. Update README badges, install instructions
  - [ ] 25. Poetry-vs-uv decision + consistent tooling
