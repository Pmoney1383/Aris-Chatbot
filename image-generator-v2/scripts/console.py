"""Terminal output shared by all threads: one live status line redrawn in
place at the bottom, with log messages printed above it.

When stdout isn't a terminal (redirected to a file), the live line is
skipped and callers fall back to periodic plain prints."""

import os
import shutil
import sys
import threading

_lock = threading.Lock()
_live = ""
IS_TTY = sys.stdout.isatty()

if IS_TTY and os.name == "nt":
    os.system("")  # turn on ANSI escape handling in older Windows consoles


def _width() -> int:
    return max(20, shutil.get_terminal_size((120, 20)).columns - 1)


def _clear_line():
    sys.stdout.write("\r\x1b[2K")


def log(msg: str):
    """Print a message above the live line (or plainly, if not a terminal)."""
    with _lock:
        if IS_TTY and _live:
            _clear_line()
        sys.stdout.write(msg + "\n")
        if IS_TTY and _live:
            sys.stdout.write(_live)
        sys.stdout.flush()


def live(line: str):
    """Replace the live status line. Truncated to the terminal width so it
    never wraps (a wrapped line can't be redrawn in place)."""
    global _live
    if not IS_TTY:
        return
    with _lock:
        _live = line[:_width()]
        _clear_line()
        sys.stdout.write(_live)
        sys.stdout.flush()


def end_live():
    """Leave the last live line on screen and move below it."""
    global _live
    with _lock:
        if IS_TTY and _live:
            sys.stdout.write("\n")
            sys.stdout.flush()
        _live = ""
