"""Textual app.

Screens live in this module, registered through SCREENS, with all styling in
acgt.tcss rather than inline. A widget used by two screens moves to a flat
widgets.py.
"""

from typing import ClassVar

from textual.app import App


class ACGTApp(App):
    """The ACGT terminal application."""

    CSS_PATH = "acgt.tcss"
    TITLE = "ACGT"
    SCREENS: ClassVar = {}


def main():
    ACGTApp().run()
