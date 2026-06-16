"""Integration tests for the full end-to-end pipeline."""

import json

# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestIngestToSynthesizeToExport:
    """Full pipeline: ingest -> synthesize (mocked) -> export."""

    def test_ingest_markdown_then_export(self, tmp_path):
        """Create a markdown file, ingest it, then export the result."""
        from distill_align.core.schemas import ConversationSchema, DataChunk, SourceMetadata, SynthesizedTurn
        from distill_align.exporter.formatters.sharegpt import ShareGPTFormatter
        from distill_align.ingestion.loaders.markdown import MarkdownLoader

        # Create a markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a test document.", encoding="utf-8")

        # Ingest
        loader = MarkdownLoader(md_file)
        content = loader.load()
        assert content is not None and len(content) > 0

        # Create a conversation from the content (simulating synthesis)
        chunk = DataChunk(
            content=content,
            metadata=SourceMetadata(
                source_type="markdown",
                file_path=str(md_file),
                file_name="test.md",
            ),
        )
        conv = ConversationSchema(
            id="integration-test-1",
            source_chunk_id=chunk.id,
            turns=[
                SynthesizedTurn(role="system", content="You are helpful."),
                SynthesizedTurn(role="user", content=chunk.content),
                SynthesizedTurn(role="assistant", content="I read the document."),
            ],
        )

        # Export to ShareGPT
        out_dir = tmp_path / "export"
        out_dir.mkdir()
        formatter = ShareGPTFormatter(out_dir)
        output = formatter.format([conv], "test_sharegpt.json")
        assert output.exists()

        # Verify the output
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["id"] == "integration-test-1"

    def test_ingestion_file_types(self, tmp_path):
        """Test that various file types can be ingested."""
        from distill_align.ingestion.loaders.csv_loader import CSVLoader
        from distill_align.ingestion.loaders.json_loader import JSONLoader
        from distill_align.ingestion.loaders.markdown import MarkdownLoader
        from distill_align.ingestion.loaders.text import TextLoader

        # Text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text content.", encoding="utf-8")
        txt_loader = TextLoader(txt_file)
        assert txt_loader.load() == "Plain text content."

        # Markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Markdown\n\nContent.", encoding="utf-8")
        md_loader = MarkdownLoader(md_file)
        assert "# Markdown" in md_loader.load()

        # JSON file
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        json_loader = JSONLoader(json_file)
        assert "key" in json_loader.load()

        # CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4", encoding="utf-8")
        csv_loader = CSVLoader(csv_file)
        assert "1" in csv_loader.load()

    def test_export_pipeline_all_formats(self, tmp_path):
        """Test export pipeline with all formats."""
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
        from distill_align.exporter.formatters.alpaca import AlpacaFormatter
        from distill_align.exporter.formatters.chatml import ChatMLFormatter
        from distill_align.exporter.formatters.conversation import ConversationFormatter
        from distill_align.exporter.formatters.sharegpt import ShareGPTFormatter

        conversations = [
            ConversationSchema(
                id=f"format-test-{i}",
                source_chunk_id=f"chunk-{i}",
                turns=[
                    SynthesizedTurn(role="user", content=f"Question {i}"),
                    SynthesizedTurn(role="assistant", content=f"Answer {i}"),
                ],
            )
            for i in range(3)
        ]

        out_dir = tmp_path / "formats"
        out_dir.mkdir()

        formatters = [
            ("sharegpt", ShareGPTFormatter(out_dir), "sharegpt.json"),
            ("alpaca", AlpacaFormatter(out_dir), "alpaca.json"),
            ("chatml", ChatMLFormatter(out_dir), "chatml.json"),
            ("conversation", ConversationFormatter(out_dir), "conv.json"),
        ]

        for name, fmt, filename in formatters:
            result = fmt.format(conversations, filename)
            assert result.exists(), f"{name} formatter produced no output"
            data = json.loads(result.read_text(encoding="utf-8"))
            assert len(data) == 3, f"{name} formatter expected 3 items, got {len(data)}"
            assert fmt.validate(data), f"{name} formatter validation failed"


class TestAutoIngestion:
    """Tests for auto-ingestion pipeline."""

    def test_auto_detect_markdown(self, tmp_path):
        """AutoIngestionPipeline detects markdown files."""
        from distill_align.core.schemas import IngestionConfig
        from distill_align.ingestion.auto import AutoIngestionPipeline

        md_file = tmp_path / "test.md"
        md_file.write_text("# Auto detect\n\nContent.", encoding="utf-8")

        pipeline = AutoIngestionPipeline(IngestionConfig())
        chunks = pipeline.ingest_file(md_file)
        assert len(chunks) > 0
        assert chunks[0].metadata.source_type == "markdown"

    def test_auto_detect_text(self, tmp_path):
        """AutoIngestionPipeline detects text files."""
        from distill_align.core.schemas import IngestionConfig
        from distill_align.ingestion.auto import AutoIngestionPipeline

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Just some notes.", encoding="utf-8")

        pipeline = AutoIngestionPipeline(IngestionConfig())
        chunks = pipeline.ingest_file(txt_file)
        assert len(chunks) > 0

    def test_auto_detect_json(self, tmp_path):
        """AutoIngestionPipeline detects JSON files."""
        from distill_align.core.schemas import IngestionConfig
        from distill_align.ingestion.auto import AutoIngestionPipeline

        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")

        pipeline = AutoIngestionPipeline(IngestionConfig())
        chunks = pipeline.ingest_file(json_file)
        assert len(chunks) > 0

    def test_auto_detect_csv(self, tmp_path):
        """AutoIngestionPipeline detects CSV files."""
        from distill_align.core.schemas import IngestionConfig
        from distill_align.ingestion.auto import AutoIngestionPipeline

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4", encoding="utf-8")

        pipeline = AutoIngestionPipeline(IngestionConfig())
        chunks = pipeline.ingest_file(csv_file)
        assert len(chunks) > 0
