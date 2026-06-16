"""
Textual TUI application for Distill-Align.

Provides an interactive dashboard for monitoring pipeline execution.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    DataTable,
    ProgressBar,
    Label,
    Select,
    Input,
    Checkbox,
)
from textual.binding import Binding
from textual import work
from textual.message import Message

from ..core.config import get_settings


class StatsPanel(Static):
    """Panel displaying pipeline statistics."""

    def compose(self) -> ComposeResult:
        yield Label("📊 Pipeline Statistics", classes="panel-title")
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Metric", "Value")
        table.add_rows(
            [
                ("Total Chunks", "0"),
                ("Conversations", "0"),
                ("Success Rate", "0%"),
                ("Processing Time", "0s"),
            ]
        )


class ControlPanel(Static):
    """Panel with pipeline controls."""

    def compose(self) -> ComposeResult:
        yield Label("⚙️ Controls", classes="panel-title")
        yield Button("▶ Start Pipeline", id="start", variant="success")
        yield Button("⏸ Pause", id="pause", variant="warning")
        yield Button("⏹ Stop", id="stop", variant="error")
        yield Button("🔄 Reset", id="reset")


class ProgressPanel(Static):
    """Panel showing pipeline progress."""

    def compose(self) -> ComposeResult:
        yield Label("📈 Progress", classes="panel-title")
        yield ProgressBar(total=100, id="main-progress")
        yield Label("Stage: Idle", id="stage-label")


class ConfigPanel(Static):
    """Panel for configuration."""

    def compose(self) -> ComposeResult:
        yield Label("🔧 Configuration", classes="panel-title")
        yield Select(
            [("OpenAI", "openai"), ("Ollama", "ollama"), ("vLLM", "vllm")],
            prompt="LLM Provider",
            id="provider-select",
        )
        yield Input(placeholder="Model name...", id="model-input")
        yield Input(placeholder="Source path...", id="source-input")
        yield Checkbox("Enable Socratic Transformer", True, id="socratic-check")
        yield Checkbox("Enable Scaffold Action", True, id="scaffold-check")


class LogPanel(Static):
    """Panel showing log messages."""

    def compose(self) -> ComposeResult:
        yield Label("📝 Logs", classes="panel-title")
        yield Static("Waiting for pipeline to start...", id="log-content")

    def add_log(self, message: str) -> None:
        """Add a log message."""
        log_content = self.query_one("#log-content", Static)
        current = log_content.renderable
        log_content.update(f"{current}\n{message}")


class DistillAlignApp(App):
    """Main TUI application."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #stats-panel {
        column-span: 1;
        row-span: 1;
        border: solid green;
        padding: 1;
    }

    #control-panel {
        column-span: 1;
        row-span: 1;
        border: solid blue;
        padding: 1;
    }

    #progress-panel {
        column-span: 1;
        row-span: 1;
        border: solid yellow;
        padding: 1;
    }

    #log-panel {
        column-span: 1;
        row-span: 1;
        border: solid red;
        padding: 1;
    }

    Button {
        margin: 1 0;
    }

    Select, Input {
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start_pipeline", "Start"),
        Binding("p", "pause_pipeline", "Pause"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatsPanel(id="stats-panel")
        yield ControlPanel(id="control-panel")
        yield ProgressPanel(id="progress-panel")
        yield LogPanel(id="log-panel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "start":
            self.action_start_pipeline()
        elif event.button.id == "pause":
            self.action_pause_pipeline()
        elif event.button.id == "stop":
            self.action_stop_pipeline()
        elif event.button.id == "reset":
            self.action_reset()

    def action_start_pipeline(self) -> None:
        """Start the pipeline."""
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log("Starting pipeline...")
        # TODO: Implement pipeline execution
        self.notify("Pipeline started!")

    def action_pause_pipeline(self) -> None:
        """Pause the pipeline."""
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log("Pipeline paused.")
        self.notify("Pipeline paused!")

    def action_stop_pipeline(self) -> None:
        """Stop the pipeline."""
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log("Pipeline stopped.")
        self.notify("Pipeline stopped!")

    def action_reset(self) -> None:
        """Reset the pipeline."""
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log("Pipeline reset.")
        self.notify("Pipeline reset!")


if __name__ == "__main__":
    app = DistillAlignApp()
    app.run()
