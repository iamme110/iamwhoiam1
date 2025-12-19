# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Reusable progress dialog for downloads
"""

from typing import Optional
from gi.repository import Gtk, Adw, GObject, GLib


class ProgressDialog(GObject.Object):
    """Progress dialog for download operations"""

    __gsignals__ = {
        'cancelled': (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self, parent, url: str, content_length: Optional[str] = None):
        super().__init__()
        self.parent = parent
        self.url = url
        self.cancelled = False

        # Create dialog
        self.dialog = Adw.AlertDialog(
            heading="Downloading Video",
            body="Downloading video from URL...",
        )

        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_show_text(True)
        self.progress_bar.set_text("Connecting...")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.append(self.progress_bar)
        self.dialog.set_extra_child(box)

        self.dialog.add_response("cancel", "Cancel")
        self.dialog.set_response_appearance("cancel", Adw.ResponseAppearance.DESTRUCTIVE)

        # Connect response handler
        def on_response(dialog, task):
            try:
                response = dialog.choose_finish(task)
                if response == "cancel":
                    self.cancelled = True
                    self.emit('cancelled')
            except GLib.Error:
                pass

        self.dialog.choose(parent, None, on_response)

    def set_status(self, text: str) -> None:
        """Update status text"""
        self.progress_bar.set_text(text)

    def update_progress(self, fraction: float, downloaded: int, total: int) -> None:
        """Update progress bar"""
        self.progress_bar.set_fraction(min(fraction, 1.0))
        downloaded_mb = downloaded // (1024 * 1024)
        total_mb = total // (1024 * 1024)
        self.progress_bar.set_text(f"{downloaded_mb} MB / {total_mb} MB")

    def close(self) -> None:
        """Close the dialog"""
        try:
            self.dialog.close()
        except Exception:
            # Dialog might already be closed
            pass