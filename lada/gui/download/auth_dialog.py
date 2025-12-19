# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Authentication dialog for FTP and other protocols
"""

from typing import Optional
from gi.repository import Gtk, Adw, GObject, GLib

from .credentials import AuthCredentials


class AuthDialog(GObject.Object):
    """Authentication dialog for protocols requiring credentials"""

    __gsignals__ = {
        'auth-provided': (GObject.SignalFlags.RUN_FIRST, None, (GObject.TYPE_PYOBJECT,)),
        'cancelled': (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.dialog = None

    def show(self) -> None:
        """Show the authentication dialog"""
        self.dialog = Adw.AlertDialog(
            heading="FTP Authentication",
            body="Enter credentials for FTP access:",
        )

        username_entry = Gtk.Entry()
        username_entry.set_placeholder_text("Username")

        password_entry = Gtk.PasswordEntry()

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.append(Gtk.Label(label="Username:"))
        box.append(username_entry)
        box.append(Gtk.Label(label="Password:"))
        box.append(password_entry)
        self.dialog.set_extra_child(box)

        def on_response(dialog, task):
            try:
                response = dialog.choose_finish(task)
                if response == "connect":
                    username = username_entry.get_text().strip()
                    password = password_entry.get_text().strip()
                    creds = AuthCredentials(username=username, password=password)
                    self.emit('auth-provided', creds)
                else:
                    self.emit('cancelled')
            except GLib.Error:
                self.emit('cancelled')

        self.dialog.add_response("cancel", "Cancel")
        self.dialog.add_response("connect", "Connect")
        self.dialog.set_response_appearance("connect", Adw.ResponseAppearance.SUGGESTED)
        self.dialog.choose(self.parent, None, on_response)