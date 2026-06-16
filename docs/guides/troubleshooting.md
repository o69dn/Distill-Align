# Troubleshooting

Common issues and their solutions when working with Distill-Align.

## Installation Issues

### Poetry install fails

**Problem**: poetry install fails with dependency resolution errors.

**Solutions**:
- Ensure you have Python 3.11 or higher: python --version
- Update Poetry: pip install --upgrade poetry
- Clear Poetry cache: poetry cache clear --all pypi
- Try installing with pip instead: pip install -e .

## Ingestion Issues

### Unsupported file format

**Problem**: UnsupportedFormatError

**Solutions**:
- Check supported extensions in the Configuration Guide
- Use --no-auto flag
- The TextLoader may work as a fallback

## Synthesis Issues

### LLM provider connection errors

**Problem**: LLMClientError or connection timeout.

**Solutions**:
- Check your API key is set
- Verify the provider URL
- Reduce concurrency: --concurrency 2

### Job interrupted mid-way

**Problem**: Synthesis job failed mid-way.

**Solutions**:
- Resume with the same job ID: distill-align synthesize --input chunks.json --job-id my-job --resume
- Check job status: distill-align jobs list

## Debug Mode

### Enable debug logging

distill-align --log-level DEBUG ingest --source ./data

### Check configuration

distill-align status
distill-align config show

## Getting Help

Search GitHub Issues, start a Discussion, or open a new issue.
