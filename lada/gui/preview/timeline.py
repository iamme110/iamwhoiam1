# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import pathlib
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from gi.repository import Gtk, GObject, Gdk, Graphene, Gsk, Adw, GLib, GdkPixbuf

from lada.gui import utils

here = pathlib.Path(__file__).parent.resolve()

@dataclass
class TimelineColors:
    timeline_color: Gdk.RGBA
    playhead_color: Gdk.RGBA
    cursor_color: Gdk.RGBA

@Gtk.Template(string=utils.translate_ui_xml(here / 'timeline.ui'))
class Timeline(Gtk.Widget):
    __gtype_name__ = 'Timeline'

    @GObject.Property(type=Adw.StyleManager)
    def style_manager(self):
        return self._style_manager

    @style_manager.setter
    def style_manager(self, value):
        self._style_manager = value

    @GObject.Property()
    def parent_widget(self):
        return self._parent_widget

    @parent_widget.setter
    def parent_widget(self, value):
        self._parent_widget = value
        # Set popover parent to the video overlay so thumbnail appears over video content
        if value and hasattr(value, 'box_video_preview'):
            self.preview_popover.set_parent(value.box_video_preview)
        elif value:
            self.preview_popover.set_parent(value)

    @GObject.Property()
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self.update_duration(value)
        self.update_playhead_position(0)

    @GObject.Property()
    def playhead_position(self):
        return self._playhead_position

    @playhead_position.setter
    def playhead_position(self, value):
        self.update_playhead_position(value)

    @GObject.Signal(name="seek_requested", arg_types=(GObject.TYPE_INT64,))
    def seek_requested_signal(self, position: int):
        pass

    @GObject.Signal(name="cursor_position_changed", arg_types=(GObject.TYPE_INT64,))
    def cursor_position(self, position: int | None):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._playhead_position = 0
        self.cursor_position_x: int | None = None
        self._duration = 0
        self._last_cursor_position_x: int | None = None
        self.set_hexpand(True)

        # Threading for non-blocking thumbnail generation
        self._thumbnail_thread = None
        self._thumbnail_lock = threading.Lock()
        self._pending_thumbnail_request = None

        self.gesture_drag = Gtk.GestureDrag.new()
        self.drag_start = 0
        def on_drag_begin(gesture_drag, x, y):
            gesture_drag.set_state( Gtk.EventSequenceState.CLAIMED)
            self.on_drag_begin(x)
        self.gesture_drag.connect("drag-begin", on_drag_begin)
        self.gesture_drag.connect("drag-end", lambda _, offset_x, offset_y: self.on_drag_end(offset_x))
        self.add_controller(self.gesture_drag)

        event_controller_motion = Gtk.EventControllerMotion.new()
        event_controller_motion.connect("leave", lambda _: self.update_cursor_position(None))
        event_controller_motion.connect("motion", lambda _, x, y: self.update_cursor_position(x))
        self.add_controller(event_controller_motion)

        self._style_manager = None
        self._parent_widget = None
        self._current_thumbnail = None

        # Create popover for thumbnail display
        self.preview_popover = Gtk.Popover.new()
        self.preview_popover.set_autohide(False)  # Prevent auto-hiding when clicking elsewhere
        self.preview_image = Gtk.Picture.new()
        self.preview_popover.set_child(self.preview_image)

    def update_duration(self, value):
        self._duration = value
        self._playhead_position = 0
        self.queue_draw()

    def update_playhead_position(self, value):
        self._playhead_position = value
        self.queue_draw()

    def on_drag_end(self, offset_x):
        x = self.drag_start + offset_x
        x = max(0, x)
        allocation = self.get_allocation()
        width = allocation.width
        new_position = int((x / width) * self._duration)
        self.update_playhead_position(new_position)
        self.emit('seek_requested', new_position)

    def on_drag_begin(self, x):
        self.drag_start = x

    def update_cursor_position(self, x):
        old_position = self.cursor_position_x
        self.cursor_position_x = x

        if x:
            allocation = self.get_allocation()
            width = allocation.width
            cursor_position = int((x / width) * self._duration)
        else:
            cursor_position = -1

        # Hide popover if no cursor position
        if cursor_position == -1:
            if self.preview_popover.get_visible():
                self.preview_popover.popdown()
            self._current_thumbnail = None
            self._last_cursor_position_x = None
            self.cursor_position_x = None
            self.queue_draw()

        self.queue_draw()
        self.emit('cursor_position_changed', cursor_position)

        seek_preview_enabled = False
        if self._parent_widget and hasattr(self._parent_widget, '_config') and self._parent_widget._config:
            seek_preview_enabled = getattr(self._parent_widget._config, 'seek_preview_enabled', False)

        # Always show cursor line when hovering, regardless of seek preview setting
        if seek_preview_enabled and cursor_position != -1:
            # Check if cursor moved significantly (at least 1px)
            should_update = (self._last_cursor_position_x is None or
                            abs(x - self._last_cursor_position_x) > 1)

            if should_update:
                self._last_cursor_position_x = x
                # Always generate new thumbnail for new position
                with self._thumbnail_lock:
                    # Signal to cancel any existing thread
                    if self._thumbnail_thread and self._thumbnail_thread.is_alive():
                        self._thumbnail_thread = None  # Signal to stop
                    self._pending_thumbnail_request = (cursor_position, int(x), 0)

                # Start new thumbnail generation thread
                self._thumbnail_thread = threading.Thread(
                    target=self._generate_thumbnail_async,
                    args=(cursor_position, int(x), 0),
                    daemon=True
                )
                self._thumbnail_thread.start()

    def do_snapshot(self, s: Gtk.Snapshot):
        """
        AFAIK, it's currently only possible to get the accent color programmatically. Other colors need apparently be to be
        hardcoded and pray that it somewhat matches the documented Adwaita colors (unless another theme is used)
        https://gnome.pages.gitlab.gnome.org/libadwaita/doc/main/css-variables.html
        """
        allocation = self.get_allocation()
        width = allocation.width
        height = allocation.height

        playhead_position_x = min(int((self._playhead_position / self._duration) * width), width - 1) if self._duration > 0 else 0

        colors = self.get_timeline_colors()

        cursor_width = 2
        playhead_width = 4
        border_radius = 10

        clip_rect = Graphene.Rect().init(0, 0, width, height)
        rounded_clip_rect = Gsk.RoundedRect()
        rounded_clip_rect.init_from_rect(clip_rect, border_radius)
        s.push_rounded_clip(rounded_clip_rect)

        background_rect = Graphene.Rect().init(0, 0, width, height)
        background_rounded_rect = Gsk.RoundedRect()
        background_rounded_rect.init_from_rect(background_rect, border_radius)
        s.push_rounded_clip(background_rounded_rect)
        s.append_color(colors.timeline_color, background_rect)
        s.pop()

        playhead_rect_x = playhead_position_x - (playhead_width // 2)
        if playhead_rect_x < 0:
            playhead_rect_x = 0
        elif playhead_rect_x + playhead_width > width:
            playhead_rect_x = width - cursor_width
        playhead_rect = Graphene.Rect().init(playhead_rect_x, 0, playhead_width, height)
        s.append_color(colors.playhead_color, playhead_rect)

        if self.cursor_position_x:
            cursor_rect_x = self.cursor_position_x - (cursor_width // 2)
            if cursor_rect_x < 0:
                cursor_rect_x = 0
            elif cursor_rect_x + cursor_width > width:
                cursor_rect_x = width - cursor_width
            cursor_rect = Graphene.Rect().init(cursor_rect_x, 0, cursor_width, height)
            s.append_color(colors.cursor_color, cursor_rect)

        s.pop()

    def get_timeline_colors(self) -> TimelineColors:
        if self._style_manager:
            playhead_color = self._style_manager.get_accent_color()
            uses_dark_scheme = bool(self._style_manager.get_dark())
        else:
            playhead_color = Adw.AccentColor.BLUE
            uses_dark_scheme = False

        # On current libadwaita==1.7.0 / PyGObject==3.52.3 Adw.AccentColor.to_rgba() takes no additional argument,
        # previously one had to pass itself
        try:
            playhead_color = playhead_color.to_rgba()
        except TypeError:
            playhead_color = playhead_color.to_rgba(playhead_color)

        timeline_color = Gdk.RGBA()
        cursor_color = Gdk.RGBA()
        if uses_dark_scheme:
            timeline_color.parse("#ffffff1a")
            cursor_color.parse("#ffffffff")
        else:
            timeline_color.parse("#0000001a")
            cursor_color.parse("#000000ff")

        return TimelineColors(timeline_color, playhead_color, cursor_color)

    def _get_thumbnail_dimensions(self):
        """Get thumbnail dimensions based on configuration"""
        size_config = 'standard'  # default
        if self._parent_widget and hasattr(self._parent_widget, '_config') and self._parent_widget._config:
            size_config = getattr(self._parent_widget._config, 'seek_preview_size', 'standard')

        if size_config == 'huge':
            return 320, 180  # 50% bigger than large
        elif size_config == 'large':
            return 240, 135  # 50% bigger than standard
        else:  # standard
            return 160, 90

    def show_preview(self, thumbnail: np.ndarray, x: int, y: int):
        if thumbnail is not None and self._parent_widget:
            self._current_thumbnail = thumbnail.copy()

            # Convert BGR to RGB for GdkPixbuf
            rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

            # Create pixbuf from bytes in memory
            height, width, channels = rgb_thumbnail.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes.new(rgb_thumbnail.tobytes()),
                GdkPixbuf.Colorspace.RGB,
                False,  # has_alpha
                8,      # bits_per_sample
                width,
                height,
                width * channels
            )

            # Set pixbuf directly to picture
            self.preview_image.set_pixbuf(pixbuf)

            # Position above the timeline
            self.preview_popover.set_position(Gtk.PositionType.TOP)

            # Calculate proper coordinates relative to the video widget
            # The timeline is in box_playback_controls, but popover parent is box_video_preview
            timeline_allocation = self.get_allocation()

            # Get timeline position relative to video widget (popover parent)
            if hasattr(self._parent_widget, 'box_playback_controls') and hasattr(self._parent_widget, 'box_video_preview'):
                # Get position of playback controls relative to video preview
                playback_pos = self._parent_widget.box_playback_controls.translate_coordinates(
                    self._parent_widget.box_video_preview, 0, 0
                )
                if playback_pos:
                    playback_x_offset = playback_pos[0]
                    playback_y_offset = playback_pos[1]

                    # Get timeline position within playback controls
                    timeline_pos = self.translate_coordinates(
                        self._parent_widget.box_playback_controls, 0, 0
                    )
                    if timeline_pos:
                        timeline_x_in_playback = timeline_pos[0]
                        timeline_y_in_playback = timeline_pos[1]

                        # Point to cursor position on timeline
                        pointing_rect = Gdk.Rectangle()
                        pointing_rect.x = int(playback_x_offset + timeline_x_in_playback + x - width // 2)
                        pointing_rect.y = int(playback_y_offset + timeline_y_in_playback - 1)  # Above timeline
                        pointing_rect.width = width
                        pointing_rect.height = 1

                        self.preview_popover.set_pointing_to(pointing_rect)
                    else:
                        # Fallback without timeline position
                        pointing_rect = Gdk.Rectangle()
                        pointing_rect.x = int(playback_x_offset + x - width // 2)
                        pointing_rect.y = int(playback_y_offset - 1)  # Above timeline
                        pointing_rect.width = width
                        pointing_rect.height = 1
                        self.preview_popover.set_pointing_to(pointing_rect)
                else:
                    # Fallback if coordinate translation fails
                    pointing_rect = Gdk.Rectangle()
                    pointing_rect.x = int(x - width // 2)
                    pointing_rect.y = -1  # Above timeline
                    pointing_rect.width = width
                    pointing_rect.height = 1
                    self.preview_popover.set_pointing_to(pointing_rect)
            else:
                # Final fallback
                pointing_rect = Gdk.Rectangle()
                pointing_rect.x = int(x - width // 2)
                pointing_rect.y = -1  # Above timeline
                pointing_rect.width = width
                pointing_rect.height = 1
                self.preview_popover.set_pointing_to(pointing_rect)

            if not self.preview_popover.get_visible():
                self.preview_popover.popup()
            else:
                # If already visible, force it to reposition
                self.preview_popover.popdown()
                self.preview_popover.popup()


    def _generate_thumbnail_async(self, timestamp_ns: int, x: int, y: int):
        """Generate thumbnail in background thread without blocking UI"""
        try:
            # Generate thumbnail using parent's method
            if self._parent_widget and hasattr(self._parent_widget, 'generate_thumbnail_for_timestamp'):
                file_path = None
                if hasattr(self._parent_widget, 'current_file') and self._parent_widget.current_file:
                    file_path = self._parent_widget.current_file.get_path()

                if file_path:
                    thumb = self._parent_widget.generate_thumbnail_for_timestamp(file_path, timestamp_ns)
                    if thumb is not None:
                        # Check again if request is still valid before showing
                        with self._thumbnail_lock:
                            if self._thumbnail_thread is not None:
                                GLib.idle_add(self.show_preview, thumb, x, y)
        except Exception as e:
            import traceback
            traceback.print_exc()
