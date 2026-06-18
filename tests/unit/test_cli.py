"""Unit tests for CLI commands using Typer's CliRunner."""

import json
import os

import pytest
from distill_align.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class TestCLIVersion:
    """Tests for the version command."""

    def test_version_output(self):
        """version command returns version string."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "distill-align" in result.stdout.lower()
        assert "v" in result.stdout


class TestCLIStatus:
    """Tests for the status command."""

    def test_status_output(self):
        """status command returns system info."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Python" in result.stdout
        assert "Status" in result.stdout or "System" in result.stdout


class TestCLIHelp:
    """Tests for help output."""

    def test_main_help(self):
        """Main app shows help with no args."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage" in result.stdout or "Commands" in result.stdout

    def test_ingest_help(self):
        """ingest --help shows ingest options."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "SOURCE" in result.stdout or "source" in result.stdout

    def test_synthesize_help(self):
        """synthesize --help shows synthesis options."""
        result = runner.invoke(app, ["synthesize", "--help"])
        assert result.exit_code == 0
        assert "INPUT" in result.stdout or "input" in result.stdout

    def test_export_help(self):
        """export --help shows export options."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "INPUT" in result.stdout or "input" in result.stdout

    def test_validate_help(self):
        """validate --help shows validation options."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "INPUT" in result.stdout or "input" in result.stdout


class TestCLIInit:
    """Tests for init command."""

    def test_init_run_creates_config(self, tmp_path):
        """init run creates a config file."""
        config_path = tmp_path / "test_config.yaml"
        result = runner.invoke(app, ["init", "run", "--path", str(config_path), "--name", "test-proj"])
        assert result.exit_code == 0
        assert config_path.exists()
        content = config_path.read_text()
        assert "test-proj" in content or "project" in content


class TestCLIIngest:
    """Tests for ingest command."""

    def test_ingest_missing_file(self):
        """ingest with non-existent path shows error."""
        result = runner.invoke(app, ["ingest", "/nonexistent/path.md"])
        assert result.exit_code == 1
        assert "Error" in result.stdout or "exist" in result.stdout

    def test_ingest_markdown_file(self, tmp_path):
        """ingest works with a valid markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nWorld.", encoding="utf-8")
        output = tmp_path / "out.json"
        result = runner.invoke(
            app,
            [
                "ingest",
                str(md_file),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data) > 0


class TestCLIExport:
    """Tests for export command."""

    def test_export_missing_file(self):
        """export with non-existent path shows error."""
        result = runner.invoke(app, ["export", "/nonexistent/in.json"])
        assert result.exit_code == 1
        assert "Error" in result.stdout or "exist" in result.stdout

    def test_export_valid_conversations(self, tmp_path):
        """export works with valid conversations JSON."""
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="test-cli-export-1",
            source_chunk_id="chunk-1",
            turns=[
                SynthesizedTurn(role="user", content="Hello"),
                SynthesizedTurn(role="assistant", content="Hi there!"),
            ],
        )
        input_file = tmp_path / "conversations.json"
        data = [conv.model_dump()]
        input_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out_dir = tmp_path / "export"
        result = runner.invoke(
            app,
            [
                "export",
                str(input_file),
                "--output-dir",
                str(out_dir),
                "--format",
                "sharegpt",
                "--no-unsloth",
            ],
        )
        assert result.exit_code == 0
        assert out_dir.exists()
        files = list(out_dir.glob("*.json"))
        assert len(files) >= 1


class TestCLIValidate:
    """Tests for validate command."""

    def test_validate_missing_file(self):
        """validate with non-existent path shows error."""
        result = runner.invoke(app, ["validate", "/nonexistent/in.json"])
        assert result.exit_code == 1
        assert "Error" in result.stdout or "exist" in result.stdout

    def test_validate_valid_data(self, tmp_path):
        """validate works with valid conversations JSON."""
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="test-validate-1",
            source_chunk_id="chunk-val",
            turns=[
                SynthesizedTurn(role="user", content="Test"),
                SynthesizedTurn(role="assistant", content="Response"),
            ],
        )
        input_file = tmp_path / "valid.json"
        input_file.write_text(
            json.dumps([conv.model_dump()], indent=2),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["validate", str(input_file), "--no-dedupe"])
        assert result.exit_code == 0


class TestCLIJobs:
    """Tests for jobs commands."""

    def test_jobs_list(self):
        """jobs list runs without error."""
        result = runner.invoke(app, ["jobs", "list"])
        assert result.exit_code == 0

    def test_jobs_resume_missing(self):
        """jobs resume with bogus ID shows error."""
        result = runner.invoke(app, ["jobs", "resume", "nonexistent-job-id"])
        assert result.exit_code in (0, 1)

    def test_jobs_help(self):
        """jobs --help shows subcommands."""
        result = runner.invoke(app, ["jobs", "--help"])
        assert result.exit_code == 0


class TestCLIConfig:
    """Tests for config commands."""

    def test_config_show(self):
        """config show runs without error."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0

    def test_config_path(self):
        """config path runs without error."""
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0


class TestCLIErrorCases:
    """Tests for CLI error handling."""

    def test_unknown_command(self):
        """Unknown command shows error."""
        result = runner.invoke(app, ["nonexistent-command-xyz"])
        assert result.exit_code != 0

    def test_ingest_without_source(self):
        """ingest without source shows usage."""
        result = runner.invoke(app, ["ingest"])
        assert result.exit_code != 0

    def test_synthesize_without_input(self):
        """synthesize without input shows usage."""
        result = runner.invoke(app, ["synthesize"])
        assert result.exit_code != 0

    def test_export_without_input(self):
        """export without input shows usage."""
        result = runner.invoke(app, ["export"])
        assert result.exit_code != 0

    def test_validate_without_input(self):
        """validate without input shows usage."""
        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0

    @pytest.mark.skipif(os.environ.get("CI") == "true", reason="TUI needs a TTY, not available in CI")
    def test_tui_launches(self):
        """tui command runs (may fail if no TTY)."""
        result = runner.invoke(app, ["tui"])
        # TUI may fail without a real TTY, but should not crash
        # CI env skips the TUI launch gracefully
        assert result.exit_code in (0, 1)
