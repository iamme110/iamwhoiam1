# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging

from gi.repository import Adw, Gtk

from lada import get_available_detection_models, LOG_LEVEL

def _(text):
    return text

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class ModelSelectionDialog:
    def __init__(self, current_model: str | None = None, parent=None):
        self.current_model = current_model
        self.parent = parent
        self.selected_model = None
        
        # Create the dialog
        self.dialog = Adw.AlertDialog(
            heading=_("Select Detection Model"),
            body=_("Choose the detection model to use for this file:")
        )
        
        # Create the content area
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(12)
        content_box.set_margin_start(12)
        content_box.set_margin_end(12)
        
        # Add label
        label = Gtk.Label(label=_("Detection Model:"))
        label.set_xalign(0)
        content_box.append(label)
        
        # Create combo box
        self.combo_box = Gtk.ComboBoxText()
        self.combo_box.set_hexpand(True)
        
        # Populate with available models
        available_models = get_available_detection_models()
        if not available_models:
            logger.warning("No detection models available")
            available_models = ['v4-fast']  # fallback
        
        for model in available_models:
            self.combo_box.append_text(model)
        
        # Set the current model as selected if provided
        if current_model and current_model in available_models:
            selected_idx = available_models.index(current_model)
            self.combo_box.set_active(selected_idx)
        elif available_models:
            self.combo_box.set_active(0)  # Select first model by default
        
        content_box.append(self.combo_box)
        
        # Set the extra child
        self.dialog.set_extra_child(content_box)
        
        # Add responses
        self.dialog.add_response("cancel", _("Cancel"))
        self.dialog.add_response("confirm", _("Apply"))
        self.dialog.set_default_response("confirm")
        
    def choose(self, callback):
        """Show the dialog and call callback with response"""
        def on_response_selected(_dialog, task):
            response = self.dialog.choose_finish(task)
            if response == "confirm":
                self.selected_model = self.combo_box.get_active_text()
            callback(_dialog, task)
        
        self.dialog.choose(self.parent, None, on_response_selected)
        
    def get_selected_model(self) -> str | None:
        """Get the currently selected detection model"""
        return self.selected_model