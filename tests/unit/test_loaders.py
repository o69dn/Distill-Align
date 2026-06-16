"""Unit tests for all file loaders."""

import json


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_text_file(self, tmp_path):
        """TextLoader loads plain text correctly."""
        from distill_align.ingestion.loaders.text import TextLoader

        content = "Hello, world!\nThis is a text file."
        f = tmp_path / "test.txt"
        f.write_text(content, encoding="utf-8")
        loader = TextLoader(f)
        result = loader.load()
        assert result == content

    def test_extract_metadata(self, tmp_path):
        """TextLoader.extract_metadata returns correct metadata."""
        from distill_align.ingestion.loaders.text import TextLoader

        f = tmp_path / "data.txt"
        f.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")
        loader = TextLoader(f)
        meta = loader.extract_metadata()
        assert meta.source_type == "text"
        assert meta.file_name == "data.txt"
        assert meta.custom_tags["lines"] == 3
        assert meta.custom_tags["format"] == "text"

    def test_unsupported_encoding_fallback(self, tmp_path):
        """TextLoader falls back to latin-1 on UnicodeDecodeError."""
        from distill_align.ingestion.loaders.text import TextLoader

        f = tmp_path / "latin1.txt"
        f.write_bytes("Caf\xe9 au lait".encode("latin-1"))
        loader = TextLoader(f)
        result = loader.load()
        assert "Caf" in result


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_load_markdown(self, tmp_path):
        """MarkdownLoader loads content correctly."""
        from distill_align.ingestion.loaders.markdown import MarkdownLoader

        content = "# Title\n\nSome content."
        f = tmp_path / "doc.md"
        f.write_text(content, encoding="utf-8")
        loader = MarkdownLoader(f)
        assert loader.load() == content

    def test_extract_headers(self, tmp_path):
        """MarkdownLoader extracts headers from content."""
        from distill_align.ingestion.loaders.markdown import MarkdownLoader

        content = """# Main Title

Intro.

## Section 1

Content.

### Subsection

Detail.

## Section 2

More."""
        f = tmp_path / "doc.md"
        f.write_text(content, encoding="utf-8")
        loader = MarkdownLoader(f)
        meta = loader.extract_metadata()
        assert meta.source_type == "markdown"
        assert "Main Title" in meta.section_headers
        assert "Section 1" in meta.section_headers
        assert "Section 2" in meta.section_headers
        assert meta.title == "Main Title"

    def test_custom_tags(self, tmp_path):
        """MarkdownLoader sets format custom tag."""
        from distill_align.ingestion.loaders.markdown import MarkdownLoader

        f = tmp_path / "readme.md"
        f.write_text("# Readme", encoding="utf-8")
        loader = MarkdownLoader(f)
        meta = loader.extract_metadata()
        assert meta.custom_tags["format"] == "markdown"


class TestCodeLoader:
    """Tests for CodeLoader."""

    def test_load_python_file(self, tmp_path):
        """CodeLoader loads Python files correctly."""
        from distill_align.ingestion.loaders.code import CodeLoader

        content = "def hello():\n    print('hi')\n"
        f = tmp_path / "hello.py"
        f.write_text(content, encoding="utf-8")
        loader = CodeLoader(f)
        assert loader.load() == content

    def test_detect_language(self, tmp_path):
        """CodeLoader detects language from extension."""
        from distill_align.ingestion.loaders.code import CodeLoader

        f = tmp_path / "main.py"
        f.write_text("x = 1", encoding="utf-8")
        loader = CodeLoader(f)
        assert loader.language == "python"

        f2 = tmp_path / "app.js"
        f2.write_text("let x = 1;", encoding="utf-8")
        loader2 = CodeLoader(f2)
        assert loader2.language == "javascript"

    def test_extract_definitions(self, tmp_path):
        """CodeLoader extracts function/class definitions."""
        from distill_align.ingestion.loaders.code import CodeLoader

        content = """def func1(): pass

def func2(x): return x

class MyClass:
    def method(self): pass
"""
        f = tmp_path / "lib.py"
        f.write_text(content, encoding="utf-8")
        loader = CodeLoader(f)
        meta = loader.extract_metadata()
        assert "func1" in meta.custom_tags["functions"]
        assert "func2" in meta.custom_tags["functions"]
        assert "MyClass" in meta.custom_tags["classes"]

    def test_code_metadata(self, tmp_path):
        """CodeLoader returns correct source_type."""
        from distill_align.ingestion.loaders.code import CodeLoader

        f = tmp_path / "module.py"
        f.write_text("x = 1", encoding="utf-8")
        loader = CodeLoader(f)
        meta = loader.extract_metadata()
        assert meta.source_type == "code"
        assert meta.language == "python"


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_load_csv(self, tmp_path):
        """CSVLoader loads CSV with headers."""
        from distill_align.ingestion.loaders.csv_loader import CSVLoader

        content = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        f = tmp_path / "data.csv"
        f.write_text(content, encoding="utf-8", newline="")
        loader = CSVLoader(f)
        result = loader.load()
        assert "Alice" in result
        assert "Bob" in result

    def test_load_csv_specific_column(self, tmp_path):
        """CSVLoader extracts specific column with text_column."""
        from distill_align.ingestion.loaders.csv_loader import CSVLoader

        content = "title,description\nHello,World\nFoo,Bar\n"
        f = tmp_path / "data.csv"
        f.write_text(content, encoding="utf-8", newline="")
        loader = CSVLoader(f, text_column="description")
        result = loader.load()
        assert "World" in result

    def test_extract_metadata(self, tmp_path):
        """CSVLoader extracts column and row metadata."""
        from distill_align.ingestion.loaders.csv_loader import CSVLoader

        content = "a,b,c\n1,2,3\n4,5,6\n"
        f = tmp_path / "data.csv"
        f.write_text(content, encoding="utf-8", newline="")
        loader = CSVLoader(f)
        meta = loader.extract_metadata()
        assert meta.custom_tags["column_count"] == 3
        assert meta.custom_tags["row_count"] == 2
        assert meta.custom_tags["format"] == "csv"


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_load_json_object(self, tmp_path):
        """JSONLoader loads JSON object."""
        from distill_align.ingestion.loaders.json_loader import JSONLoader

        data = {"key": "value", "num": 42}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(f)
        result = loader.load()
        assert "key" in result
        assert "value" in result

    def test_load_json_array(self, tmp_path):
        """JSONLoader loads JSON array."""
        from distill_align.ingestion.loaders.json_loader import JSONLoader

        data = [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(f)
        result = loader.load()
        assert "Entry 1" in result
        assert "Entry 2" in result

    def test_load_jsonl(self, tmp_path):
        """JSONLoader handles JSONL format."""
        from distill_align.ingestion.loaders.json_loader import JSONLoader

        lines = [json.dumps({"text": f"entry {i}"}) for i in range(3)]
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(lines), encoding="utf-8")
        loader = JSONLoader(f)
        result = loader.load()
        assert "Entry 1" in result
        assert "entry 0" in result

    def test_extract_metadata(self, tmp_path):
        """JSONLoader extracts entry count metadata."""
        from distill_align.ingestion.loaders.json_loader import JSONLoader

        data = [{"a": 1}, {"a": 2}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        loader = JSONLoader(f)
        meta = loader.extract_metadata()
        assert meta.custom_tags["entries"] == 2
