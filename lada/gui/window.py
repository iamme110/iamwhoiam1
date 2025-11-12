# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
import threading
from math import sqrt

from gi.repository import Adw, Gtk, Gio, GLib, GObject, Gdk

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.export.export_view import ExportView
from lada.gui.fileselection.file_selection_view import FileSelectionView
from lada.gui.preview.preview_view import PreviewView
from lada.gui.shortcuts import ShortcutsManager

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'window.ui'))
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    file_selection_view: FileSelectionView = Gtk.Template.Child()
    export_view: ExportView = Gtk.Template.Child()
    preview_view: PreviewView = Gtk.Template.Child()
    view_stack: Adw.ViewStack = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value
        self._setup_shortcuts()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._config: Config | None
        self._shortcuts_manager: ShortcutsManager | None = None

        self.set_title("Lada")

        self.connect("close-request", self.close)
        self.file_selection_view.connect("files-selected", lambda obj, files: self.on_files_selected(files))
        self.preview_view.connect("toggle-fullscreen-requested", lambda *args: self.on_toggle_fullscreen())
        self.preview_view.connect("window-resize-requested", self.on_window_resize_requested)
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

        self.export_view.props.view_stack = self.view_stack
        self.export_view.connect("video-export-requested", lambda obj, restore_directory_or_file: self.on_video_export_requested(restore_directory_or_file))
        self.preview_view.props.view_stack = self.view_stack

    def on_video_export_requested(self, restore_directory_or_file: Gio.File):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "export"
        def run():
            self.preview_view.close(block=True)
            GLib.idle_add(lambda: self.export_view.start_export(restore_directory_or_file))
        threading.Thread(target=run).start()

    def on_files_selected(self, files: list[Gio.File]):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "preview" if self._config.initial_view == "preview" else "export"
        self.preview_view.add_files(files)
        if self.view_stack.props.visible_child_name == "preview":
            self.preview_view.play_file(0)
        self.export_view.add_files(files)

    def on_fullscreened(self, fullscreened: bool):
        if self.stack.props.visible_child_name == "main" and self.view_stack.props.visible_child_name == "preview":
            self.preview_view.on_fullscreened(fullscreened)

    def on_toggle_fullscreen(self):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def on_window_resize_requested(self, obj, paintable: Gdk.Paintable, playback_controls: Gtk.Widget, header_bar: Gtk.Widget):
        logger.debug("Window resize requested from config toggle")
        if self.is_visible():
            self._resize_window(paintable, playback_controls, header_bar)
        else:
            self.connect("map", self._resize_window, paintable, playback_controls, header_bar, True)

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("ui", "UI")
        def switch_views(child_name):
            if self.stack.props.visible_child_name == "main":
                self.view_stack.set_visible_child_name(child_name)
        self._shortcuts_manager.add("ui", "show-export-view", "e", lambda *args: switch_views('export'), _("Switch to Export View"))
        self._shortcuts_manager.add("ui", "show-preview-view", "p", lambda *args: switch_views('preview'), _("Switch to Watch View"))

    def close(self, *args):
        self.preview_view.close()
        self.export_view.close()

    def _resize_window(self, paintable: Gdk.Paintable, playback_controls: Gtk.Widget, headerbar: Gtk.Widget, initial: bool | None = False, video_width: int = 0, video_height: int = 0) -> None:
        # Check if window centering is enabled - if not, skip all resizing logic
        if not hasattr(self, '_config') or not hasattr(self._config, 'enable_window_centering') or not self._config.enable_window_centering:
            logger.debug("Window centering disabled, skipping resize")
            return

        # Copied from https://gitlab.gnome.org/GNOME/showtime/-/blob/3c940ff2a4128a50c559985a04fb6beb7e9292e6/showtime/widgets/window.py
        # SPDX-License-Identifier: GPL-3.0-or-later
        # SPDX-FileCopyrightText: Copyright 2024-2025 kramo

        # For large enough monitors, occupy 40% of the screen area when opening a window with a video
        DEFAULT_OCCUPY_SCREEN = 0.4

        # Screens with this resolution or smaller are handled as small
        SMALL_SCREEN_AREA = 1280 * 1024

        # For small monitors, occupy 80% of the screen area
        SMALL_OCCUPY_SCREEN = 0.8

        SMALL_SIZE_CHANGE = 10

        logger.debug("Resizing window…")

        if initial:
            self.disconnect_by_func(self._resize_window)

        # Check if paintable is None or invalid
        if paintable is None:
            logger.debug("Paintable is None, skipping resize")
            return

        # Use real video dimensions if provided, otherwise fallback to paintable dimensions
        if video_width == 0 or video_height == 0:
            if not (video_width := paintable.get_intrinsic_width()) or not (
                    video_height := paintable.get_intrinsic_height()
            ):
                logger.debug("Paintable has invalid dimensions, skipping resize")
                return

        if not (surface := self.get_surface()):
            logger.error("Could not get GdkSurface to resize window")
            return

        if not (monitor := self.props.display.get_monitor_at_surface(surface)):
            logger.error("Could not get GdkMonitor to resize window")
            return

        # Log display and video information
        monitor_rect = monitor.props.geometry
        logger.debug(f"Display monitor: {monitor_rect.width}x{monitor_rect.height} at ({monitor_rect.x}, {monitor_rect.y})")
        logger.debug(f"Video resolution: {video_width}x{video_height}")

        video_area = video_width * video_height
        init_width, init_height = self.get_default_size()
        logger.debug(f"Window current size: {init_width}x{init_height}")

        playback_controls_height, _natural, _minimum_baseline, _natural_baseline = playback_controls.measure(Gtk.Orientation.VERTICAL, video_height)
        header_bar_height, _natural, _minimum_baseline, _natural_baseline = headerbar.measure(Gtk.Orientation.VERTICAL, video_height)
        additional_height_needed_for_controls = playback_controls_height + header_bar_height

        # Calculate space needed for timeline preview popover
        timeline_preview_space = 0
        if hasattr(self, '_config') and hasattr(self._config, 'seek_preview_enabled') and self.is_maximized() and self._config.seek_preview_enabled:
            # Add space for thumbnail preview (approx 90px height + margin)
            timeline_preview_space = 120

        if initial:
            # Algorithm copied from Loupe
            # https://gitlab.gnome.org/GNOME/loupe/-/blob/4ca5f9e03d18667db5d72325597cebc02887777a/src/widgets/image/rendering.rs#L151

            hidpi_scale = surface.props.scale_factor

            monitor_rect = monitor.props.geometry

            monitor_width = monitor_rect.width
            monitor_height = monitor_rect.height

            monitor_area = monitor_width * monitor_height
            logical_monitor_area = monitor_area * pow(hidpi_scale, 2)

            occupy_area_factor = (
                SMALL_OCCUPY_SCREEN
                if logical_monitor_area <= SMALL_SCREEN_AREA
                else DEFAULT_OCCUPY_SCREEN
            )

            if hasattr(self, '_config') and hasattr(self._config, 'adjust_video_size_to_resolution') and self._config.adjust_video_size_to_resolution:
                target_scale = 1.0
            else:
                size_scale = sqrt(monitor_area / video_area * occupy_area_factor)
                target_scale = min(1, size_scale)

            nat_width = video_width * target_scale
            nat_height = video_height * target_scale

            # margin is estimated space for Dock or Taskbar. In some OS these can also be placed left/right of the monitor so use it for both width/height
            margin = 100
            max_width = monitor_width - margin * hidpi_scale
            original_nat_width = nat_width
            original_nat_height = nat_height
            if nat_width > max_width:
                nat_width = max_width
                nat_height = video_height * nat_width / video_width

            max_height = monitor_height - margin * hidpi_scale - timeline_preview_space
            if nat_height > max_height:
                nat_height = max_height
                nat_width = video_width * nat_height / video_height

            if hasattr(self, '_config') and hasattr(self._config, 'adjust_video_size_to_resolution') and self._config.adjust_video_size_to_resolution and (nat_width < original_nat_width or nat_height < original_nat_height):
                self.maximize()
                return

        else:
            # For subsequent video changes, use the same sizing logic as initial
            # but check if adjust_video_size_to_resolution is enabled
            hidpi_scale = surface.props.scale_factor

            monitor_rect = monitor.props.geometry

            monitor_width = monitor_rect.width
            monitor_height = monitor_rect.height

            monitor_area = monitor_width * monitor_height
            logical_monitor_area = monitor_area * pow(hidpi_scale, 2)

            occupy_area_factor = (
                SMALL_OCCUPY_SCREEN
                if logical_monitor_area <= SMALL_SCREEN_AREA
                else DEFAULT_OCCUPY_SCREEN
            )

            adjust_to_resolution = hasattr(self, '_config') and hasattr(self._config, 'adjust_video_size_to_resolution') and self._config.adjust_video_size_to_resolution

            if adjust_to_resolution:
                # Exact video resolution mode
                nat_width = video_width
                nat_height = video_height
            else:
                # Scale to fit monitor while maintaining aspect ratio
                size_scale = sqrt(monitor_area / video_area * occupy_area_factor)
                target_scale = min(1, size_scale)

                nat_width = video_width * target_scale
                nat_height = video_height * target_scale

            # Check if the calculated window size would exceed screen bounds and maximize if needed
            # This is independent of the adjust_video_size_to_resolution setting - if the window would be too big, maximize
            available_width = monitor_rect.width
            available_height = monitor_rect.height

            # On Windows, account for taskbar
            import platform
            if platform.system() == 'Windows':
                try:
                    import ctypes
                    user32 = ctypes.windll.user32
                    work_area = ctypes.wintypes.RECT()
                    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0)  # SPI_GETWORKAREA
                    available_width = work_area.right - work_area.left
                    available_height = work_area.bottom - work_area.top
                    logger.debug(f"Windows taskbar detected, using work area {available_width}x{available_height} instead of monitor {monitor_rect.width}x{monitor_rect.height}")
                except Exception as taskbar_e:
                    logger.debug(f"Could not get Windows work area: {taskbar_e}, using full monitor dimensions")

            # Check if the video resolution would create a window larger than available screen space
            # This triggers maximization when videos are too large for the screen
            margin_pixels = 100 * hidpi_scale
            exact_video_width_needed = video_width + margin_pixels
            exact_video_height_needed = video_height + margin_pixels + additional_height_needed_for_controls + timeline_preview_space

            logger.debug(f"Calculated video size: {nat_width}x{nat_height}, exact video would need: {exact_video_width_needed}x{exact_video_height_needed}")
            logger.debug(f"Available space: {available_width}x{available_height}")

            # If window is currently maximized but new video doesn't need maximization, unmaximize it
            if self.is_maximized() and not (exact_video_width_needed > available_width or exact_video_height_needed > available_height):
                logger.debug(f"Unmaximizing window: previous video required maximization, but new video fits in available space")
                self.unmaximize()

            # Maximize if the exact video resolution would exceed screen bounds
            if exact_video_width_needed > available_width or exact_video_height_needed > available_height:
                logger.debug(f"Maximizing window: exact video size ({exact_video_width_needed}x{exact_video_height_needed}) exceeds available space ({available_width}x{available_height})")
                self.maximize()
                return

            # Check if change is significant enough to resize
            if (abs(init_width - nat_width) < SMALL_SIZE_CHANGE) and (
                    abs(init_height - nat_height) < SMALL_SIZE_CHANGE
            ):
                return

        # Apply margin constraints
        hidpi_scale = surface.props.scale_factor
        monitor_rect = monitor.props.geometry
        monitor_width = monitor_rect.width
        monitor_height = monitor_rect.height
        margin = 100
        max_width = monitor_width - margin * hidpi_scale
        max_height = monitor_height - margin * hidpi_scale - timeline_preview_space

        # Store original calculated size before applying constraints
        original_nat_width = nat_width
        original_nat_height = nat_height

        # Apply constraints
        if nat_width > max_width:
            nat_width = max_width
            nat_height = video_height * nat_width / video_width

        if nat_height > max_height:
            nat_height = max_height
            nat_width = video_width * nat_height / video_height

        # Check if we need to maximize due to size constraints
        # This applies both when adjust_video_size_to_resolution is enabled AND when the window would be too large
        size_too_large = (original_nat_width > monitor_width - margin * hidpi_scale) or (original_nat_height > monitor_height - margin * hidpi_scale - timeline_preview_space)
        adjust_to_resolution = hasattr(self, '_config') and hasattr(self._config, 'adjust_video_size_to_resolution') and self._config.adjust_video_size_to_resolution

        if adjust_to_resolution and (nat_width < original_nat_width or nat_height < original_nat_height or size_too_large):
            logger.debug(f"Maximizing window: adjust_to_resolution={adjust_to_resolution}, size_too_large={size_too_large}, constrained=({nat_width}x{nat_height}), original=({original_nat_width}x{original_nat_height})")
            self.maximize()
            return

        nat_width = round(nat_width)
        nat_height = round(nat_height) + additional_height_needed_for_controls

        logger.debug(f"Window new size calculated: {nat_width}x{nat_height}")

        # Always center the window on screen after resize for better UX (if enabled)
        if not initial and not self.is_maximized():
            if not (surface := self.get_surface()):
                logger.error("Could not get GdkSurface to center window")
            else:
                monitor = self.props.display.get_monitor_at_surface(surface)
                if monitor:
                    monitor_rect = monitor.props.geometry
                    logger.debug(f"Centering window on monitor: {monitor_rect.width}x{monitor_rect.height} at ({monitor_rect.x}, {monitor_rect.y})")

                    # Calculate center position
                    center_x = monitor_rect.x + (monitor_rect.width - nat_width) // 2

                    # For Windows, account for taskbar by adjusting available height
                    import platform
                    centering_enabled = hasattr(self, '_config') and hasattr(self._config, 'enable_window_centering') and self._config.enable_window_centering

                    if centering_enabled and platform.system() == 'Windows':
                        try:
                            import ctypes
                            user32 = ctypes.windll.user32
                            # Get work area (excluding taskbar)
                            work_area = ctypes.wintypes.RECT()
                            user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0)  # SPI_GETWORKAREA
                            available_height = work_area.bottom - work_area.top
                            center_y = work_area.top + (available_height - nat_height) // 2
                            logger.debug(f"Windows taskbar detected, using work area height {available_height} instead of monitor height {monitor_rect.height}")
                        except Exception as taskbar_e:
                            logger.debug(f"Could not get Windows work area: {taskbar_e}, using full monitor height")
                            center_y = monitor_rect.y + (monitor_rect.height - nat_height) // 2
                    elif centering_enabled:
                        center_y = monitor_rect.y + (monitor_rect.height - nat_height) // 2
                    else:
                        # Centering disabled - just present window without moving
                        self.set_property("default-width", nat_width)
                        self.set_property("default-height", nat_height)
                        self.present()
                        return

                    # Ensure window stays within monitor bounds
                    center_x = max(monitor_rect.x, min(center_x, monitor_rect.x + monitor_rect.width - nat_width))
                    center_y = max(monitor_rect.y, min(center_y, monitor_rect.y + monitor_rect.height - nat_height))
                    logger.debug(f"Adjusted center position (bounds checked): ({center_x}, {center_y})")

                    # Set size and position
                    self.set_property("default-width", nat_width)
                    self.set_property("default-height", nat_height)

                    def reposition_window():
                        try:
                            # Calculate center position again (same logic as above)
                            if platform.system() == 'Windows':
                                try:
                                    import ctypes
                                    user32 = ctypes.windll.user32
                                    work_area = ctypes.wintypes.RECT()
                                    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0)
                                    available_height = work_area.bottom - work_area.top
                                    center_y = work_area.top + (available_height - nat_height) // 2
                                except Exception as taskbar_e:
                                    center_y = monitor_rect.y + (monitor_rect.height - nat_height) // 2
                            else:
                                center_y = monitor_rect.y + (monitor_rect.height - nat_height) // 2

                            center_x = monitor_rect.x + (monitor_rect.width - nat_width) // 2
                            center_x = max(monitor_rect.x, min(center_x, monitor_rect.x + monitor_rect.width - nat_width))
                            center_y = max(monitor_rect.y, min(center_y, monitor_rect.y + monitor_rect.height - nat_height))

                            # Try different methods to move the window (prefer more reliable methods first)
                            window_moved = False

                            # Method 1: Try GTK4 positioning using Win32 API for Windows
                            try:
                                if platform.system() == 'Windows':
                                    # For Windows with GTK4, we need to use the native Win32 window handle
                                    native = self.get_native()
                                    if native:
                                        surface = native.get_surface()
                                        if surface:
                                            # On Windows, surface is GdkWin32Toplevel, try to get the window handle
                                            logger.debug(f"Surface type {type(surface)} - checking for Win32 positioning")

                                            # Try to use Win32 API directly if available
                                            try:
                                                import ctypes
                                                import ctypes.wintypes

                                                # Get the window handle (HWND) from the surface
                                                # For GdkWin32Toplevel, we can access the handle
                                                hwnd = None
                                                try:
                                                    # Try different ways to get the handle
                                                    if hasattr(surface, 'get_handle'):
                                                        hwnd = surface.get_handle()
                                                    elif hasattr(surface, 'get_hwnd'):
                                                        hwnd = surface.get_hwnd()
                                                    else:
                                                        # Try to access the handle through GObject properties
                                                        hwnd = surface.get_property('handle') if hasattr(surface, 'get_property') else None
                                                except:
                                                    pass

                                                if not hwnd:
                                                    # Fallback: Try FindWindow with window title
                                                    try:
                                                        import ctypes
                                                        import ctypes.wintypes
                                                        import os
                                                        window_title = self.get_title()
                                                        if window_title is None:
                                                            window_title = "Lada"  # Replace with your app name
                                                        user32 = ctypes.windll.user32
                                                        # First try FindWindowW with exact title
                                                        hwnd = user32.FindWindowW(None, window_title)
                                                        if not hwnd and window_title:
                                                            # Try FindWindowA with ASCII title
                                                            try:
                                                                hwnd = user32.FindWindowA(None, window_title.encode('utf-8'))
                                                            except:
                                                                pass
                                                        if not hwnd:
                                                            # Try finding window by class name "gdkWindowToplevel"
                                                            try:
                                                                hwnd = user32.FindWindowA(b"gdkWindowToplevel", None)
                                                            except:
                                                                pass
                                                        if not hwnd:
                                                            # Try finding any window with our process ID using EnumWindows
                                                            pid = os.getpid()
                                                            found_hwnd = None
                                                            def enum_windows_callback(hwnd, lParam):
                                                                if lParam[0] == 0:
                                                                    pid_buffer = ctypes.wintypes.DWORD()
                                                                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buffer))
                                                                    if pid_buffer.value == lParam[1]:
                                                                        lParam[0] = hwnd
                                                                return True
                                                            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
                                                            callback = EnumWindowsProc(enum_windows_callback)
                                                            lParam = [0, pid]
                                                            user32.EnumWindows(callback, ctypes.byref(ctypes.c_int(pid)))
                                                            found_hwnd = lParam[0]
                                                            if found_hwnd:
                                                                hwnd = found_hwnd

                                                    except Exception as find_e:
                                                        logger.debug(f"FindWindow fallback failed: {find_e}")

                                                if hwnd:
                                                    logger.debug(f"Got Win32 window handle: {hwnd}")

                                                    # Use Windows API to move the window
                                                    user32 = ctypes.windll.user32
                                                    # SWP_NOSIZE (0x0001) | SWP_NOZORDER (0x0004) = 0x0005
                                                    result = user32.SetWindowPos(hwnd, 0, center_x, center_y, 0, 0, 0x0005)
                                                    if result:
                                                        window_moved = True
                                                        logger.debug(f"Window moved using Win32 SetWindowPos to ({center_x}, {center_y})")
                                                    else:
                                                        logger.debug("Win32 SetWindowPos returned False")
                                                else:
                                                    logger.debug("Could not get window handle from surface")
                                            except Exception as win32_e:
                                                logger.debug(f"Win32 positioning failed: {win32_e}")

                                            if not window_moved and hasattr(surface, 'set_position'):
                                                surface.set_position(center_x, center_y)
                                                window_moved = True
                                                logger.debug(f"Window moved using surface.set_position() to ({center_x}, {center_y})")
                                            elif not window_moved:
                                                logger.debug(f"Surface type {type(surface)} doesn't support positioning methods")
                                        else:
                                            logger.debug("Could not get surface from native")
                                    else:
                                        logger.debug("Could not get native window")
                            except Exception as e1:
                                logger.debug(f"GTK4 positioning failed: {e1}")

                            # Method 2: Try setting position property
                            if not window_moved:
                                try:
                                    # Try to set position on the native window
                                    if hasattr(self, 'get_native'):
                                        native = self.get_native()
                                        if native and hasattr(native, 'set_position'):
                                            native.set_position(center_x, center_y)
                                            window_moved = True
                                            logger.debug(f"Window moved using native.set_position() to ({center_x}, {center_y})")
                                        else:
                                            logger.debug("Native window doesn't have set_position method")
                                    else:
                                        logger.debug("No get_native method available for set_position")

                                    if not window_moved:
                                        # Fallback to trying set_position on self
                                        try:
                                            self.set_position(center_x, center_y)
                                            window_moved = True
                                            logger.debug(f"Window moved using self.set_position() to ({center_x}, {center_y})")
                                        except:
                                            pass

                                    # Get actual position after set_position to verify
                                    try:
                                        if hasattr(self, 'get_position'):
                                            actual_x, actual_y = self.get_position()
                                            logger.debug(f"Actual window position after set_position: ({actual_x}, {actual_y})")
                                        else:
                                            logger.debug("get_position method not available")
                                    except Exception as e_pos:
                                        logger.debug(f"Could not get actual window position after set_position: {e_pos}")
                                except Exception as e2:
                                    logger.debug(f"All set_position methods failed: {e2}")

                            # Method 3: Try accessing underlying surface or toplevel
                            if not window_moved:
                                try:
                                    # Try to get the native toplevel
                                    toplevel = self.get_native() if hasattr(self, 'get_native') else None
                                    if toplevel:
                                        # For GTK4, try setting default position before showing
                                        self.set_default_size(nat_width, nat_height)
                                        # Force re-realization of the window
                                        self.unrealize() if hasattr(self, 'unrealize') else None
                                        self.map() if hasattr(self, 'map') else None
                                        logger.debug(f"Window unrealize/map attempted for repositioning")
                                except Exception as e3:
                                    logger.debug(f"Native toplevel repositioning failed: {e3}")

                            # Method 4: Try setting window hints for centering
                            if not window_moved:
                                try:
                                    # Set window positioning hints that might help the WM center it
                                    self.set_property('resizable', True)
                                    # Force window manager attention
                                    self.present()
                                    logger.debug(f"Set centering hints and presented window")
                                except Exception as e4:
                                    logger.debug(f"Window hints failed: {e4}")

                            # Force window update and bring to front
                            self.present()

                            if window_moved:
                                logger.debug(f"Successfully centered window to ({center_x}, {center_y}) for video resolution adjustment")
                            else:
                                logger.debug(f"Could not move window to ({center_x}, {center_y}), but presenting window")
                                # Final attempt to get current position
                                try:
                                    if hasattr(self, 'get_position'):
                                        current_x, current_y = self.get_position()
                                        logger.debug(f"Final window position: ({current_x}, {current_y})")
                                    else:
                                        logger.debug("get_position method not available for final position check")
                                except Exception as e_final:
                                    logger.debug(f"Could not get final window position: {e_final}")
                        except Exception as e:
                            logger.debug(f"Could not reposition window: {e}")
                            # Fallback: just present the window
                            self.present()
                        return False

                    GLib.idle_add(reposition_window)
                else:
                    # Fallback to setting properties normally
                    for prop, init, target in (
                            ("default-width", init_width, nat_width),
                            ("default-height", init_height, nat_height),
                    ):
                        self.set_property(prop, target)
        else:
            for prop, init, target in (
                    ("default-width", init_width, nat_width),
                    ("default-height", init_height, nat_height),
            ):
                anim = Adw.TimedAnimation.new(
                    self, init, target, 500, Adw.PropertyAnimationTarget.new(self, prop)
                )
                anim.props.easing = Adw.Easing.EASE_OUT_EXPO
                (anim.skip if initial else anim.play)()
                logger.debug("Resized window to %ix%i", nat_width, nat_height)