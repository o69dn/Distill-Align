"""
Textual TUI application for Distill-Align.

Provides an interactive dashboard for monitoring pipeline execution.
Features real-time stats, job management, cache inspection, and log viewing.
"""

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    TabbedContent,
    TabPane,
    RichLog,
)
from textual.binding import Binding
from textual.worker import Worker
from rich.text import Text
from rich.panel import Panel as RichPanel
from loguru import logger

from ..core.config_file import find_config_file, load_config
from ..core.checkpoint import CheckpointManager, JobStatus
from ..core.cache import CacheManager


# =============================================================================
# Dashboard Tab
# =============================================================================

class DashboardTab(Container):
    """Main dashboard showing real-time stats and progress."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📊 Pipeline Dashboard", classes="tab-title")

            with Horizontal(id="stats-row"):
                yield Static(self._stat_box("📦 Chunks", "0", "green"), id="stat-chunks", classes="stat-box")
                yield Static(self._stat_box("💬 Conversations", "0", "blue"), id="stat-conv", classes="stat-box")
                yield Static(self._stat_box("💰 Cost", "$0.00", "yellow"), id="stat-cost", classes="stat-box")
                yield Static(self._stat_box("🎯 Quality", "100%", "magenta"), id="stat-quality", classes="stat-box")

            yield ProgressBar(total=100, id="dashboard-progress", show_eta=True)
            yield Label("Stage: Idle", id="stage-label")

            with Horizontal(id="quick-stats"):
                yield DataTable(id="cache-table")

    def _stat_box(self, label: str, value: str, color: str) -> str:
        return f"[bold {color}]{label}[/bold {color}]\n[white]{value}[/white]"

    def on_mount(self) -> None:
        # Populate cache table
        table = self.query_one("#cache-table", DataTable)
        table.add_columns("Cache Metric", "Value")
        self.refresh_cache_stats()

    def refresh_cache_stats(self):
        """Refresh cache statistics in the dashboard."""
        try:
            cache = CacheManager(cache_dir=".cache")
            stats = cache.stats()
            table = self.query_one("#cache-table", DataTable)
            table.clear()
            table.add_rows([
                ("Total Entries", str(stats.total_entries)),
                ("Hits", str(stats.hit_count)),
                ("Misses", str(stats.miss_count)),
                ("Hit Rate", f"{stats.hit_rate:.1%}"),
                ("DB Size", f"{stats.db_size_mb} MB"),
            ])
        except Exception as e:
            logger.error(f"Failed to refresh cache stats: {e}")


# =============================================================================
# Jobs Tab
# =============================================================================

class JobsTab(Container):
    """Tab for managing synthesis jobs."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📋 Job Management", classes="tab-title")

            with Horizontal(id="jobs-actions"):
                yield Button("🔄 Refresh", id="jobs-refresh", variant="primary")
                yield Button("🗑️ Delete Selected", id="jobs-delete", variant="error")
                yield Button("📜 List All", id="jobs-list")

            yield DataTable(id="jobs-table")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("Job ID", "Type", "Status", "Progress", "Created", "Errors")
        self.refresh_jobs()

    def refresh_jobs(self):
        """Refresh the jobs table."""
        try:
            manager = CheckpointManager()
            jobs = manager.list_jobs(limit=50)
            table = self.query_one("#jobs-table", DataTable)
            table.clear()

            for job in jobs:
                created = datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M")
                progress = f"{job.processed_items}/{job.total_items} ({job.progress_pct:.0f}%)"
                errors = str(job.failed_items)
                table.add_row(
                    job.job_id,
                    job.job_type,
                    job.status.value,
                    progress,
                    created,
                    errors,
                )
        except Exception as e:
            logger.error(f"Failed to refresh jobs: {e}")


# =============================================================================
# Cache Tab
# =============================================================================

class CacheTab(Container):
    """Tab for inspecting the synthesis cache."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("💾 Cache Inspector", classes="tab-title")

            with Horizontal(id="cache-actions"):
                yield Button("🔄 Refresh", id="cache-refresh", variant="primary")
                yield Button("🧹 Prune Old", id="cache-prune", variant="warning")
                yield Button("🗑️ Clear All", id="cache-clear", variant="error")

            yield DataTable(id="cache-detail-table")

    def on_mount(self) -> None:
        table = self.query_one("#cache-detail-table", DataTable)
        table.add_columns("Key (prefix)", "Model", "Tokens", "Created", "Accessed")
        self.refresh_cache()

    def refresh_cache(self):
        """Refresh cache detail table."""
        try:
            cache = CacheManager(cache_dir=".cache")
            with cache._get_conn() if hasattr(cache, "_get_conn") else _no_op():
                keys = cache.get_cached_keys()

            table = self.query_one("#cache-detail-table", DataTable)
            table.clear()

            for key in keys[:100]:  # Limit display
                entry = cache.get(key)
                if entry:
                    created = datetime.fromtimestamp(entry["created_at"]).strftime("%Y-%m-%d %H:%M")
                    table.add_row(
                        key[:16] + "...",
                        entry.get("model", "unknown"),
                        str(entry.get("tokens_used", 0)),
                        created,
                        "-",
                    )
        except Exception as e:
            logger.error(f"Failed to refresh cache: {e}")


class _no_op:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


# =============================================================================
# Config Tab
# =============================================================================

class ConfigTab(Container):
    """Tab for viewing configuration."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("🔧 Configuration", classes="tab-title")
            yield Static("Loading configuration...", id="config-content")

    def on_mount(self) -> None:
        self.refresh_config()

    def refresh_config(self):
        """Refresh config display."""
        config_path = find_config_file()
        content_widget = self.query_one("#config-content", Static)

        if config_path:
            try:
                config = load_config(config_path)
                text = f"[bold green]Config File:[/bold green] {config_path}\n\n"
                text += f"[cyan]{config.model_dump_json(indent=2)}[/cyan]"
                content_widget.update(text)
            except Exception as e:
                content_widget.update(f"[red]Error loading config: {e}[/red]")
        else:
            content_widget.update(
                "[yellow]No config file found.[/yellow]\n\n"
                "Run [cyan]distill-align init[/cyan] to create one."
            )


# =============================================================================
# Logs Tab
# =============================================================================

class LogsTab(Container):
    """Tab for viewing live logs."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📝 Live Logs", classes="tab-title")
            yield RichLog(id="log-view", highlight=True, markup=True, wrap=True)


# =============================================================================
# Main App
# =============================================================================

class DistillAlignApp(App):
    """Main TUI application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    .tab-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .stat-box {
        width: 1fr;
        height: 5;
        content-align: center middle;
        border: round $primary;
        margin: 0 1;
    }

    TabbedContent {
        height: 1fr;
    }

    #cache-table {
        height: 1fr;
    }

    Button {
        margin: 0 1;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield DashboardTab()
            with TabPane("Jobs", id="jobs"):
                yield JobsTab()
            with TabPane("Cache", id="cache"):
                yield CacheTab()
            with TabPane("Config", id="config"):
                yield ConfigTab()
            with TabPane("Logs", id="logs"):
                yield LogsTab()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self.title = "Distill-Align Dashboard"
        self.sub_title = "Structured Reasoning Extraction Factory"

        # Refresh stats periodically
        self.set_interval(5.0, self.auto_refresh)

        # Add a welcome log message
        log_view = self.query_one("#log-view", RichLog)
        if log_view:
            log_view.write("[green]✓[/green] Distill-Align TUI started")
            log_view.write(f"[cyan]i[/cyan] Working directory: {Path.cwd()}")
            log_view.write(f"[cyan]i[/cyan] Press [yellow]q[/yellow] to quit, [yellow]r[/yellow] to refresh")

    def auto_refresh(self) -> None:
        """Periodic auto-refresh of stats."""
        try:
            dashboard = self.query_one(DashboardTab)
            dashboard.refresh_cache_stats()
        except Exception:
            pass

    def action_refresh(self) -> None:
        """Manual refresh action."""
        try:
            dashboard = self.query_one(DashboardTab)
            dashboard.refresh_cache_stats()
            self.notify("Refreshed stats", timeout=2)
        except Exception as e:
            self.notify(f"Refresh error: {e}", severity="error")

        # Also refresh jobs
        try:
            jobs = self.query_one(JobsTab)
            jobs.refresh_jobs()
        except Exception:
            pass

        # And config
        try:
            config = self.query_one(ConfigTab)
            config.refresh_config()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses across all tabs."""
        if event.button.id == "jobs-refresh":
            self.query_one(JobsTab).refresh_jobs()
        elif event.button.id == "jobs-list":
            self.notify("Listing all jobs...")
        elif event.button.id == "cache-refresh":
            self.query_one(CacheTab).refresh_cache()
        elif event.button.id == "cache-prune":
            cache = CacheManager(cache_dir=".cache")
            removed = cache.prune()
            self.notify(f"Pruned {removed} entries")
            self.query_one(CacheTab).refresh_cache()
        elif event.button.id == "cache-clear":
            cache = CacheManager(cache_dir=".cache")
            removed = cache.clear()
            self.notify(f"Cleared {removed} entries", severity="warning")
            self.query_one(CacheTab).refresh_cache()


if __name__ == "__main__":
    app = DistillAlignApp()
    app.run()
