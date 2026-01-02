# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
from gi.repository import Adw, Gtk, GObject, GLib
from gettext import gettext as _

logger = logging.getLogger(__name__)

class UrlInputDialog(Adw.ApplicationWindow):
    __gtype_name__ = 'UrlInputDialog'
    
    def __init__(self, parent=None, **kwargs):
        super().__init__(**kwargs)
        self._parent = parent
        self._url_text = ""
        
        # Set up the window
        self.set_title("Watch from URL")
        self.set_default_size(400, 150)
        self.set_resizable(False)
        
        # Create main content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        main_box.set_margin_top(20)
        main_box.set_margin_bottom(20)
        main_box.set_margin_start(20)
        main_box.set_margin_end(20)
        
        # Add title label
        title_label = Gtk.Label()
        title_label.set_text("Enter video URL from supported sites")
        title_label.set_halign(Gtk.Align.START)
        main_box.append(title_label)
        
        # Create URL entry
        self._url_entry = Gtk.Entry()
        self._url_entry.set_placeholder_text("https://example.com/video")
        self._url_entry.set_activates_default(True)
        self._url_entry.connect("changed", self._on_url_changed)
        main_box.append(self._url_entry)
        
        # Create button box
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        button_box.set_halign(Gtk.Align.END)
        
        # Cancel button
        cancel_button = Gtk.Button.new_with_label("Cancel")
        cancel_button.connect("clicked", self._on_cancel_clicked)
        button_box.append(cancel_button)
        
        # Download button
        self._download_button = Gtk.Button.new_with_label("Download")
        self._download_button.add_css_class("suggested-action")
        self._download_button.set_sensitive(False)
        self._download_button.connect("clicked", self._on_download_clicked)
        button_box.append(self._download_button)
        
        main_box.append(button_box)
        
        # Set content
        self.set_content(main_box)
        
        # Set up the window as transient for the parent
        if parent:
            try:
                self.set_transient_for(parent)
            except:
                pass
        

    
    def _on_url_changed(self, entry):
        """Enable download button when URL is provided"""
        url = entry.get_text().strip()
        self._download_button.set_sensitive(bool(url))
    
    def _on_cancel_clicked(self, button):
        """Cancel and dismiss dialog"""
        try:
            self.emit("dialog-dismissed")
            self.close()
        except Exception as e:
            logger.error(f"Error canceling dialog: {e}")
    
    def _on_download_clicked(self, button):
        """Confirm and emit URL"""
        try:
            url = self._url_entry.get_text().strip()
            if url:
                self.emit("url-confirmed", url)
                self.close()
        except Exception as e:
            logger.error(f"Error processing download: {e}")
    
    @GObject.Signal(name="url-confirmed", arg_types=(str,))
    def url_confirmed_signal(self, url: str):
        pass
    
    @GObject.Signal(name="dialog-dismissed")
    def dialog_dismissed_signal(self):
        pass