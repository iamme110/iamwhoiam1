# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib

from gi.repository import Adw, Gtk, Gio, GObject, GLib
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.utils import dump_encoder_options
from lada.utils import video_utils

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'encoding_preset_dialog.ui'))
class EncodingPresetDialog(Adw.Dialog):
    __gtype_name__ = 'EncodingPresetDialog'

    text_view_encoder_options: Gtk.TextView = Gtk.Template.Child()
    drop_down_encoders: Gtk.DropDown = Gtk.Template.Child()
    input_text: Gtk.Entry = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.set_follows_content_size(True)
        self.set_content_width(700)
        self.set_content_height(400)

        default_encoder = "libx264"

        self.text_view_encoder_options.set_vexpand(True)
        self.text_view_encoder_options.set_hexpand(True)
        self.text_view_encoder_options.set_monospace(True)
        self.update_text_view_encoder_options(default_encoder)

        self.encoders = video_utils.get_available_video_encoder_codecs()
        strings = Gtk.StringList()
        self.drop_down_encoders.props.model = strings
        for i, encoder in enumerate(self.encoders):
            name = f"{encoder.name} ({encoder.long_name}){f" [{" ".join(encoder.hardware_devices)}]" if len(encoder.hardware_devices) > 0 else ""}"
            strings.append(name)
            if default_encoder == encoder.name:
                self.drop_down_encoders.set_selected(i)
        self.drop_down_encoders.connect("notify::selected-item", self.on_encoder_selected)


    def on_encoder_selected(self, dropdown, _pspec):
        selected_encoder = dropdown.props.selected_item
        if selected_encoder is not None:
            idx = self.drop_down_encoders.props.model.find(selected_encoder.props.string)
            encoder = self.encoders[idx]
            self.update_text_view_encoder_options(encoder.name)

    def update_text_view_encoder_options(self, encoder: str):
        buffer = self.text_view_encoder_options.get_buffer()
        buffer.set_text(dump_encoder_options(encoder) + "\n")

    @Gtk.Template.Callback()
    def button_create_clicked_callback(self, button: Gtk.Button):
        encoder_options = self.input_text.get_text()
        print(f"create clicked. options: {encoder_options}")
        self.close()
