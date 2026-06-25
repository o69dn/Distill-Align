"""Keyboard-shortcut help overlay for the Distill-Align TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

_HELP_CONTENT = """
[bold yellow]⌨️  Keyboard Shortcuts[/bold yellow]

[bold]Navigation[/bold]
  [reverse] 1 [/reverse]  Dashboard      [reverse] 6 [/reverse]  Full Pipeline
  [reverse] 2 [/reverse]  Ingest          [reverse] 7 [/reverse]  Jobs
  [reverse] 3 [/reverse]  Synthesize      [reverse] 8 [/reverse]  Cache
  [reverse] 4 [/reverse]  Export          [reverse] 9 [/reverse]  Config
  [reverse] 5 [/reverse]  Validate        [reverse] 0 [/reverse]  Logs

[bold]Actions[/bold]
  [reverse] m [/reverse]  Toggle Simple/Expert mode
  [reverse] r [/reverse]  Refresh all stats
  [reverse] ? [/reverse]  Show this help
  [reverse] q [/reverse]  Quit application

[bold]Tips[/bold]
  • In [green]Simple mode[/green], only essential fields are shown.
  • Switch to [yellow]Expert mode[/yellow] ([reverse] m [/reverse]) for full control.
  • Use the Quick-Start Wizard on the Dashboard to get going fast.
  • Templates auto-fill sensible defaults — try one!

[dim]Press any key to close this help.[/dim]
"""


class HelpOverlay(ModalScreen[None]):
    """Modal overlay showing keyboard shortcuts and tips."""

    CSS = """
    HelpOverlay {
        align: center middle;
    }

    HelpOverlay > Static {
        width: 56;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $background;
        color: $text;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(_HELP_CONTENT)

    def on_key(self, _event: None) -> None:
        """Close on any key."""
        self.app.pop_screen()
