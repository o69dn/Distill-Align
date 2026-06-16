"""Performance benchmarks for key operations."""

import pytest

pytestmark = pytest.mark.benchmark


class TestIngestionBenchmarks:
    """Benchmarks for ingestion pipeline."""

    def test_markdown_loading(self, benchmark):
        """Benchmark MarkdownLoader file reading."""
        import tempfile
        from pathlib import Path

        from distill_align.ingestion.loaders.markdown import MarkdownLoader

        content = "# Test\n\n" + "Line of text.\n" * 1000
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp = f.name
        try:
            loader = MarkdownLoader(Path(tmp))
            result = benchmark(loader.load)
            assert len(result) > 5000
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_chunking_large_markdown(self, benchmark, sample_markdown_content):
        """Benchmark chunking on large markdown."""
        from distill_align.ingestion.chunkers.markdown import MarkdownChunker

        large = sample_markdown_content * 50
        chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=200)
        from distill_align.core.schemas import SourceMetadata

        metadata = SourceMetadata(
            source_type="markdown",
            file_path="/test/large.md",
            file_name="large.md",
        )
        chunks = benchmark(chunker.chunk, large, metadata)
        assert len(chunks) >= 5


class TestExportBenchmarks:
    """Benchmarks for export formatters."""

    def test_sharegpt_formatting(self, benchmark, sample_chunks):
        """Benchmark ShareGPT formatting."""
        import tempfile

        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
        from distill_align.exporter.formatters.sharegpt import ShareGPTFormatter

        conversations = []
        for i, chunk in enumerate(sample_chunks):
            conv = ConversationSchema(
                id=f"bench-{i}",
                source_chunk_id=chunk.id,
                turns=[
                    SynthesizedTurn(role="user", content=chunk.content),
                    SynthesizedTurn(role="assistant", content=f"Response to: {chunk.content}"),
                ],
            )
            conversations.append(conv)

        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = ShareGPTFormatter(tmpdir)
            result = benchmark(formatter.format, conversations, "bench_sharegpt.json")
            assert result.exists()

    def test_alpaca_formatting(self, benchmark, sample_chunks):
        """Benchmark Alpaca formatting."""
        import tempfile

        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
        from distill_align.exporter.formatters.alpaca import AlpacaFormatter

        conversations = []
        for i, chunk in enumerate(sample_chunks):
            conv = ConversationSchema(
                id=f"bench-{i}",
                source_chunk_id=chunk.id,
                turns=[
                    SynthesizedTurn(role="user", content=chunk.content),
                    SynthesizedTurn(role="assistant", content=f"Response to: {chunk.content}"),
                ],
            )
            conversations.append(conv)

        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = AlpacaFormatter(tmpdir)
            result = benchmark(formatter.format, conversations, "bench_alpaca.json")
            assert result.exists()


class TestSchemaBenchmarks:
    """Benchmarks for schema serialization."""

    def test_data_chunk_serialization(self, benchmark, sample_chunks):
        """Benchmark DataChunk model_dump."""
        result = benchmark(lambda: [c.model_dump() for c in sample_chunks])
        assert len(result) == len(sample_chunks)

    def test_data_chunk_deserialization(self, benchmark, sample_chunks):
        """Benchmark DataChunk deserialization."""
        from distill_align.core.schemas import DataChunk

        data = [c.model_dump() for c in sample_chunks]
        result = benchmark(lambda: [DataChunk(**d) for d in data])
        assert len(result) == len(data)
