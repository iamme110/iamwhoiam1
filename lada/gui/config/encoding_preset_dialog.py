# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib

from gi.repository import Adw, Gtk, Gio, GObject, GLib
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.utils import dump_encoder_options
from lada.utils import video_utils
from lada.utils.video_utils import EncodingPreset, Encoder

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'encoding_preset_dialog.ui'))
class EncodingPresetDialog(Adw.Dialog):
    __gtype_name__ = 'EncodingPresetDialog'

    text_view_encoder_options: Gtk.TextView = Gtk.Template.Child()
    drop_down_encoders: Gtk.DropDown = Gtk.Template.Child()
    entry_encoder_options: Gtk.Entry = Gtk.Template.Child()
    entry_description: Gtk.Entry = Gtk.Template.Child()

    def __init__(self, preset: EncodingPreset, **kwargs):
        super().__init__(**kwargs)
        # self.set_follows_content_size(True)
        self.set_content_width(700)
        self.set_content_height(400)

        self._encoding_preset = preset

        self.entry_description.set_text(self._encoding_preset.description)

        self.text_view_encoder_options.set_vexpand(True)
        self.text_view_encoder_options.set_hexpand(True)
        self.text_view_encoder_options.set_monospace(True)
        self.update_text_view_encoder_options(self._encoding_preset.encoder_name)

        self.encoders = video_utils.get_available_video_encoder_codecs()
        strings = Gtk.StringList()
        self.drop_down_encoders.props.model = strings
        for i, encoder in enumerate(self.encoders):
            name = f"{encoder.name} ({encoder.long_name}){f" [{" ".join(encoder.hardware_devices)}]" if len(encoder.hardware_devices) > 0 else ""}"
            strings.append(name)
            if self._encoding_preset.encoder_name == encoder.name:
                self.drop_down_encoders.set_selected(i)
        self.drop_down_encoders.connect("notify::selected-item", self.on_encoder_selected)

    @GObject.Property()
    def encoding_preset(self) -> EncodingPreset:
        return self._encoding_preset

    @encoding_preset.setter
    def encoding_preset(self, value: EncodingPreset):
        self._encoding_preset = value

    @GObject.Signal(name="preset-changed", arg_types=(GObject.TYPE_PYOBJECT,))
    def preset_changed_signal(self, encoding_preset: EncodingPreset):
        pass

    def on_encoder_selected(self, dropdown, _pspec):
        if selected_encoder := self.get_selected_encoder():
            self.update_text_view_encoder_options(selected_encoder.name)

    def get_selected_encoder(self) -> Encoder | None:
        selected_encoder = self.drop_down_encoders.props.selected_item
        if selected_encoder is not None:
            idx = self.drop_down_encoders.props.model.find(selected_encoder.props.string)
            return self.encoders[idx]
        return None

    def update_text_view_encoder_options(self, encoder: str):
        buffer = self.text_view_encoder_options.get_buffer()
        buffer.set_text(dump_encoder_options(encoder) + "\n")

    @Gtk.Template.Callback()
    def button_create_clicked_callback(self, button: Gtk.Button):
        description = self.entry_description.get_text()
        encoder_options = self.entry_encoder_options.get_text()
        encoder = self.get_selected_encoder()
        if encoder is not None and encoder_options is not None and description is not None and utils.is_unique_preset_description(description):
            self._encoding_preset.description = description
            self._encoding_preset.encoder_name = encoder.name
            self._encoding_preset.encoder_options = encoder_options
            self.emit("preset-changed", self.encoding_preset)
            self.close()
