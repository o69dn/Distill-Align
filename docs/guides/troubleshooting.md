# Troubleshooting

Common issues and their solutions when working with Distill-Align.

## Installation

### `pip install distill-align` fails

**Cause:** Python version too old or pip not updated.

```bash
# Check Python version (needs 3.11+)
python --version

# Update pip
pip install --upgrade pip

# Try again
pip install distill-align
```

### Poetry dependency resolution fails

**Cause:** Conflicting dependency versions.

```bash
# Clear Poetry cache
poetry cache clear --all pypi

# Regenerate lock file
poetry lock

# Install fresh
poetry install
```

### `ModuleNotFoundError: No module named 'distill_align'`

**Cause:** Package not installed in the current environment.

```bash
# If using pip
pip install -e .

# If using Poetry
poetry install

# Verify installation
distill-align version
```

## Ingestion

### `UnsupportedFormatError` for a file

**Cause:** The file extension is not recognized by any loader.

```bash
# Check supported extensions
distill-align status

# Force text loading for unknown files
# (add the extension to .env or config)
DISTILL_TEXT_EXTENSIONS=.log,.custom
```

Supported extensions: `.md`, `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.pdf`, `.docx`, `.html`, `.htm`, `.json`, `.csv`, `.ipynb`, `.yaml`, `.yml`, `.toml`, `.txt`, `.cfg`, and many more.

### Ingestion produces 0 chunks

**Cause:** Files are empty, too small, or chunk size is larger than file content.

```bash
# Use smaller chunk size
distill-align ingest --source ./data --output chunks.json --chunk-size 500

# Check file contents
wc -c ./data/*
```

### PDF ingestion is slow or fails

**Cause:** Large PDFs with complex layouts.

```bash
# Use smaller chunk size for PDFs
distill-align ingest --source ./docs --output chunks.json --chunk-size 800

# Pre-convert PDFs to text if possible
```

## Synthesis

### Connection timeout or refused

**Cause:** API endpoint unreachable or wrong base URL.

```bash
# Test connectivity
curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"

# For local Ollama, ensure it's running
ollama serve
curl http://localhost:11434/api/tags

# Check your config
distill-align config show
```

### `AuthenticationError` or 401 response

**Cause:** Invalid or missing API key.

```bash
# Verify the key is set
echo $OPENAI_API_KEY

# Test the key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Re-export if needed
export OPENAI_API_KEY=sk-...
```

### Rate limit errors (HTTP 429)

**Cause:** Too many requests per minute.

```bash
# Reduce concurrency and RPM
distill-align synthesize \
    --input chunks.json \
    --concurrency 2 \
    --rpm 30 \
    --output conversations.json

# The built-in retry logic will handle transient 429s
```

### Job interrupted mid-way

**Cause:** Network failure, Ctrl+C, or process killed.

```bash
# List available jobs
distill-align jobs list

# Resume from last checkpoint
distill-align synthesize --input chunks.json --job-id my-job --resume

# Delete a stale job
distill-align jobs delete --job-id my-job
```

### Synthesis produces very few conversations

**Cause:** Strict pruning or low-quality chunks.

```bash
# Check what's being pruned
distill-align --log-level DEBUG synthesize --input chunks.json

# Use a more capable model
distill-align synthesize --input chunks.json --model gpt-4o

# Try a different mode
distill-align synthesize --input chunks.json --mode teach
```

### `max_tokens` reached (truncated output)

**Cause:** Server-side token limit hit before response completes.

```bash
# Increase max tokens
distill-align synthesize \
    --input chunks.json \
    --max-tokens 8192 \
    --output conversations.json
```

## Export

### `ModuleNotFoundError: No module named 'pyarrow'`

**Cause:** Parquet export requires the `pyarrow` package.

```bash
pip install distill-align[parquet]
# or
pip install pyarrow
```

### Export produces empty files

**Cause:** Conversations failed validation or were all duplicates.

```bash
# Check validation first
distill-align validate --input conversations.json

# If quality score is low, check synthesis output
distill-align validate --input conversations.json --output report.json
```

### Split files have uneven sizes

**Cause:** Small dataset with few conversations.

This is expected behavior. With fewer than 20 conversations, some splits may be empty. Use `--split` only when you have enough data (50+ conversations recommended).

## Cache Issues

### Cache is stale after changing modes

```bash
# Clear the cache
rm -rf .cache/

# Or use --no-cache to skip cache entirely
distill-align synthesize --input chunks.json --no-cache
```

### Cache directory growing too large

```bash
# Check cache size
du -sh .cache/

# Prune old entries (older than 30 days)
distill-align jobs cleanup --days 30

# Clear everything
rm -rf .cache/
```

## TUI Issues

### TUI doesn't render properly

**Cause:** Terminal doesn't support required features.

```bash
# Ensure a modern terminal (iTerm2, Windows Terminal, Alacritty, etc.)
# Minimum terminal size: 80x24

# Try with a simpler terminal
TERM=xterm-256color distill-align tui
```

### TUI shows "Not a TTY"

**Cause:** Running in a non-interactive shell or pipe.

This is expected. The TUI requires an interactive terminal. Use CLI commands instead:

```bash
distill-align status
distill-align jobs list
```

## Docker Issues

### Permission denied on mounted volumes

```bash
# Ensure the container user can access the mounted directories
chmod -R 777 ./data ./output

# Or run with your user ID
docker run --user $(id -u):$(id -g) -v $(pwd)/data:/app/data distill-align
```

### Container can't connect to host Ollama

```bash
# Use host network mode
docker run --network host distill-align \
    synthesize --input chunks.json --provider ollama --base-url http://localhost:11434

# Or use Docker Compose with the local profile
docker compose --profile local up
```

## Debugging

### Enable verbose logging

```bash
distill-align --log-level DEBUG ingest --source ./data
distill-align --log-level DEBUG synthesize --input chunks.json
```

### Check system status

```bash
distill-align status
distill-align config show
```

### Inspect a specific conversation

```bash
# Use Python to examine the output
python -c "
import json
with open('conversations.json') as f:
    data = json.load(f)
for i, conv in enumerate(data[:3]):
    print(f'--- Conversation {i+1} ---')
    for turn in conv.get('turns', conv.get('conversations', [])):
        role = turn.get('role', turn.get('from', '?'))
        content = turn.get('content', turn.get('value', ''))[:200]
        print(f'{role}: {content}...')
    print()
"
```

## Getting Help

- **GitHub Issues:** [github.com/omargargoum/Distill-Align/issues](https://github.com/omargargoum/Distill-Align/issues)
- **Discussions:** [github.com/omargargoum/Distill-Align/discussions](https://github.com/omargargoum/Distill-Align/discussions)
- **Website:** [omar.com.ly](https://omar.com.ly/)
