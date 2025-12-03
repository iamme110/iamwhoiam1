# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import pathlib
import threading
import time
import traceback

from fractions import Fraction
from gi.repository import Gtk, GObject, Gio, Adw, GLib

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config, PostExportAction
from lada.gui.config.no_gpu_banner import NoGpuBanner
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemData, ExportItemDataProgress, ExportItemState
from lada.gui.export.export_multiple_files_page import ExportMultipleFilesPage
from lada.gui.export.export_single_file_page import ExportSingleFileStatusPage
from lada.gui.export.export_utils import ResumeInformation
from lada.gui.export.shutdown_manager import ShutdownManager, ShutdownError
from lada.gui.export.spinner_button import SpinnerButton
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER
from lada.utils import audio_utils, video_utils

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_view.ui'))
class ExportView(Gtk.Widget):
    __gtype_name__ = 'ExportView'

    single_file_page: ExportSingleFileStatusPage = Gtk.Template.Child()
    multiple_files_page: ExportMultipleFilesPage = Gtk.Template.Child()
    button_start_export: Gtk.Button = Gtk.Template.Child()
    button_cancel_export: SpinnerButton = Gtk.Template.Child()
    button_resume_export: SpinnerButton = Gtk.Template.Child()
    button_pause_export: SpinnerButton = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    view_switcher: Adw.ViewSwitcher = Gtk.Template.Child()
    config_sidebar = Gtk.Template.Child()
    button_add_files: Gtk.Button = Gtk.Template.Child()
    banner_no_gpu: NoGpuBanner = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._view_stack: Adw.ViewStack | None = None
        self._config: Config | None = None
        self.in_progress_idx: int | None = None
        self.single_file = True
        self.stop_requested = False
        self.pause_requested = False
        self.resume_info: ResumeInformation | None = None
        self.video_writer: video_utils.VideoWriter | None = None
        self.progress_calculator: export_utils.ProgressCalculator | None = None

        self.connect("video-export-finished", self.on_video_export_finished)
        self.connect("video-export-failed", self.on_video_export_failed)
        self.connect("video-export-progress", self.on_video_export_progress)
        self.connect("video-export-resumed", self.on_video_export_resumed)
        self.connect("video-export-paused", self.on_video_export_paused)
        self.connect("video-export-stopped", self.on_video_export_stopped)

        self.model =  Gio.ListStore(item_type=ExportItemData)
        self.multiple_files_page.bind(self.model)

        def on_files_added(obj, files):
            self.button_add_files.set_sensitive(True)
            self.add_files(files)
        self.connect("files-added", on_files_added)

        self.single_file_page.connect("start-export-requested", lambda page, button: self.on_button_start_export_clicked(button))
        self.single_file_page.connect("stop-export-requested", self.on_button_cancel_export_clicked)
        self.single_file_page.connect("pause-export-requested", self.on_button_pause_export_clicked)
        self.single_file_page.connect("resume-export-requested", self.on_button_resume_export_clicked)

        self.multiple_files_page.connect("show-error-requested", self.on_show_error_requested)
        self.multiple_files_page.connect("remove-item-requested", self.on_remove_item_requested)

        drop_target = utils.create_video_files_drop_target(lambda files: self.emit("files-added", files))
        self.add_controller(drop_target)

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        self._config.connect("notify::export-directory", self.on_config_changed)
        self._config.connect("notify::file-name-pattern", self.on_config_changed)
        self.set_restore_button_label()

    @GObject.Property(type=Adw.ViewStack)
    def view_stack(self):
        return self._view_stack

    @view_stack.setter
    def view_stack(self, value: Adw.ViewStack):
        self._view_stack = value
        def on_visible_child_name_changed(object, spec):
            visible_child_name = object.get_property(spec.name)
            if visible_child_name == "export":
                self.config_sidebar.init_sidebar_from_config(self._config)
        self._view_stack.connect("notify::visible-child-name", on_visible_child_name_changed)

    def add_files(self, added_files: list[Gio.File]):
        assert len(added_files) > 0

        for original_file in added_files:
            if any([original_file.get_path() == item.original_file.get_path() for item in self.model]):
                # duplicate
                continue
            if self._config.export_directory:
                restored_file = self.get_restored_file_path(original_file, self._config.export_directory)
            else:
                # We don't know the output directory yet. This guess needs to be updated after the user set one via FilePicker
                restored_file = self.get_restored_file_path(original_file, added_files[0].get_parent().get_path())
            export_item = ExportItemData(original_file, restored_file)
            self.model.append(export_item)

        self.single_file = len(self.model) == 1

        if self.single_file:
            self.stack.set_visible_child_name("single-file")
            self.single_file_page.on_add_file(self.model[0])
        else:
            self.stack.set_visible_child_name("multiple-files")
            self.update_export_buttons()

    def update_export_buttons(self):
        count_queued_items = sum([item.state == ExportItemState.QUEUED for item in self.model])
        is_in_progress = self.in_progress_idx is not None
        is_paused = self.resume_info is not None
        is_any_queued_items = count_queued_items > 0

        # Only show header bar buttons in multiple files mode
        # Single file mode has its own buttons in the page
        show_header_buttons = not self.single_file

        self.button_start_export.set_visible(show_header_buttons and not is_in_progress and is_any_queued_items)
        self.button_pause_export.set_visible(show_header_buttons and is_in_progress and not is_paused)
        self.button_resume_export.set_visible(show_header_buttons and is_paused)
        self.button_cancel_export.set_visible(show_header_buttons and is_in_progress)

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-failed", arg_types=(GObject.TYPE_STRING,))
    def video_export_failed_signal(self, error_message: str):
        pass

    @GObject.Signal(name="video-export-paused",)
    def video_export_paused_signal(self):
        pass

    @GObject.Signal(name="video-export-resumed",)
    def video_export_resumed_signal(self):
        pass

    @GObject.Signal(name="video-export-stopped",)
    def video_export_stopped_signal(self):
        pass

    @GObject.Signal(name="video-export-progress", arg_types=(ExportItemDataProgress,))
    def video_export_progress_signal(self, progress):
        pass

    @GObject.Signal(name="video-export-requested")
    def video_export_requested_signal(self, save_file: Gio.File):
        pass

    @GObject.Signal(name="files-added", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass

    @GObject.Signal(name="shutdown-confirmation-requested")
    def shutdown_confirmation_requested(self):
        pass

    @Gtk.Template.Callback()
    def on_button_start_export_clicked(self, start_export_button: Gtk.Button):
        logger.info("=== START EXPORT BUTTON CLICKED ===")
        logger.info(f"Export directory configured: {self._config.export_directory is not None}")
        if self._config.export_directory:
            item = self.model[self.get_next_queued_item_idx()]
            logger.info(f"Emitting video-export-requested for item: {item.original_file.get_path()}")
            self.emit("video-export-requested", item.restored_file)
        else:
            logger.info("No export directory configured, showing export dialog")
            start_export_button.set_sensitive(False)
            dismissed_callback = lambda *args: start_export_button.set_sensitive(True)
            self.show_export_dialog(dismissed_callback)

    @Gtk.Template.Callback()
    def button_add_files_callback(self, button_clicked):
        self.button_add_files.set_sensitive(False)
        callback = lambda files: self.emit("files-added", files)
        dismissed_callback = lambda *args: self.button_add_files.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    @Gtk.Template.Callback()
    def on_button_cancel_export_clicked(self, button_clicked):
        self.stop_requested = True
        self.button_pause_export.set_sensitive(False)
        self.button_cancel_export.set_sensitive(False)
        self.button_cancel_export.set_spinner_visible(True)

    @Gtk.Template.Callback()
    def on_button_pause_export_clicked(self, button_clicked):
        logger.info("Pause button clicked - setting pause_requested = True")
        assert self.resume_info is None
        self.pause_requested = True
        self.button_pause_export.set_sensitive(False)
        self.button_pause_export.set_spinner_visible(True)
        self.button_cancel_export.set_sensitive(False)

    @Gtk.Template.Callback()
    def on_button_resume_export_clicked(self, button_clicked):
        logger.info(f"Resume button clicked - resume_info exists: {self.resume_info is not None}")
        if self.resume_info is not None:
            logger.info(f"Resume info: frame_num={self.resume_info.frame_num}, frame_pts={self.resume_info.frame_pts}")

        self.button_resume_export.set_sensitive(False)
        self.button_resume_export.set_spinner_visible(True)
        self.button_cancel_export.set_sensitive(False)

        self.pause_requested = False
        assert self.in_progress_idx is not None
        item = self.model[self.in_progress_idx]
        self._start_export(item.original_file, item.restored_file)

    def on_show_error_requested(self, obj, idx):
        model_item = self.model[idx]
        export_utils.open_error_dialog(self, model_item.original_file.get_basename(), model_item.error_details)

    def on_remove_item_requested(self, obj, idx):
        self.model.remove(idx)
        self.update_export_buttons()

    def on_config_changed(self, *args):
        if self._config.export_directory:
            for idx, model_item in enumerate(self.model):
                if model_item.state == ExportItemState.QUEUED:
                    restored_file = self.get_restored_file_path(model_item.original_file, self._config.export_directory)
                    model_item.restored_file = restored_file
                    self.multiple_files_page.on_restored_file_changed(idx, restored_file)
        self.set_restore_button_label()

    def set_restore_button_label(self):
        label = _("Restore") if self._config.export_directory else _("Restore…")
        self.single_file_page.set_button_start_restore_label(label)
        self.button_start_export.set_label(label)

    def get_next_queued_item_idx(self) -> int | None:
        for idx, item in enumerate(self.model):
            if item.state == ExportItemState.QUEUED:
                return idx
        return None

    def continue_next_file(self):
        next_idx = self.get_next_queued_item_idx()
        if next_idx is None:
            # done, all queued items processed
            self.view_switcher.set_sensitive(True)
            self.config_sidebar.set_property("disabled", False)
            self.in_progress_idx = None
            self.update_export_buttons()
            self.execute_post_export_action()
        else:
            # continue, queued items remaining
            self._start_export(self.model[next_idx].original_file, self.model[next_idx].restored_file)

    def show_video_export_started(self, save_file: Gio.File):
        self.view_switcher.set_sensitive(False)
        self.config_sidebar.set_property("disabled", True)

        idx = self.get_next_queued_item_idx()
        if idx is None:
            return

        self.in_progress_idx = idx
        self.update_export_buttons()

        model_item = self.model[idx]
        model_item.state = ExportItemState.PROCESSING

        if self.single_file:
            self.single_file_page.show_video_export_started(save_file)
        self.multiple_files_page.show_video_export_started(idx)

    def on_video_export_finished(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.progress.complete()
        model_item.state = ExportItemState.FINISHED

        if self.single_file:
            self.single_file_page.on_video_export_finished()
        self.multiple_files_page.on_video_export_finished(self.in_progress_idx)

        self.continue_next_file()

    def on_video_export_progress(self, obj, progress: ExportItemDataProgress):
        if self.in_progress_idx is None:
            return

        model_item = self.model[self.in_progress_idx]
        model_item.progress = progress

        if self.single_file:
            self.single_file_page.on_video_export_progress(progress)
        self.multiple_files_page.on_video_export_progress(self.in_progress_idx, progress)

    def on_video_export_stopped(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.state = ExportItemState.QUEUED
        model_item.progress = ExportItemDataProgress()

        if self.single_file:
            self.single_file_page.on_video_export_stopped()
        self.multiple_files_page.on_video_export_stopped(self.in_progress_idx)

        self.in_progress_idx = None
        self.stop_requested = False
        self.update_export_buttons()
        self.view_switcher.set_sensitive(True)
        self.config_sidebar.set_property("disabled", False)
        self.button_start_export.set_sensitive(True)
        self.button_cancel_export.set_sensitive(True)
        self.button_cancel_export.set_spinner_visible(False)
        self.button_pause_export.set_sensitive(True)

    def on_video_export_paused(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.state = ExportItemState.PAUSED

        if self.single_file:
            self.single_file_page.on_video_export_paused()
        self.multiple_files_page.on_video_export_paused(self.in_progress_idx)

        self.update_export_buttons()
        self.button_pause_export.set_sensitive(True)
        self.button_pause_export.set_spinner_visible(False)
        self.button_cancel_export.set_sensitive(True)
        self.pause_requested = False

    def on_video_export_resumed(self, obj):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        # For normal processing resume, the state might already be PROCESSING
        # Only change from PAUSED to PROCESSING if it was actually paused
        if model_item.state == ExportItemState.PAUSED:
            model_item.state = ExportItemState.PROCESSING
        elif model_item.state != ExportItemState.PROCESSING:
            logger.warning(f"Unexpected state during resume: {model_item.state}, expected PAUSED or PROCESSING")

        if self.single_file:
            self.single_file_page.on_video_export_resumed()
        self.multiple_files_page.on_video_export_resumed(self.in_progress_idx)

        self.update_export_buttons()
        self.button_resume_export.set_sensitive(True)
        self.button_resume_export.set_spinner_visible(False)
        self.button_cancel_export.set_sensitive(True)

    def _ensure_resume_state_transition(self):
        """Ensure model item state is correctly set to PROCESSING during resume."""
        assert self.in_progress_idx is not None
        model_item = self.model[self.in_progress_idx]
        if model_item.state == ExportItemState.PAUSED:
            model_item.state = ExportItemState.PROCESSING

    def on_video_export_failed(self, obj, error_message):
        assert self.in_progress_idx is not None

        model_item = self.model[self.in_progress_idx]
        model_item.state = ExportItemState.FAILED
        model_item.error_details = error_message

        if self.single_file:
            self.single_file_page.on_video_export_failed()
        self.multiple_files_page.on_video_export_failed(self.in_progress_idx)

        export_utils.open_error_dialog(self, model_item.original_file.get_basename(), error_message)

        self.continue_next_file()

    def start_export(self, restore_directory_or_file: Gio.File):
        logger.info("=== START_EXPORT METHOD CALLED ===")
        logger.info(f"Restore directory/file: {restore_directory_or_file.get_path()}")
        logger.info(f"Export directory configured: {self._config.export_directory is not None}")

        # Update initial guessed output restore directory/file now that the user has provided it via file/dir picker dialog
        if not self._config.export_directory:
            logger.info("No export directory configured, updating restored files")
            restored_files: list[Gio.File] = []
            if self.single_file:
                assert len(self.model) == 1
                restored_file = restore_directory_or_file
                model_item = self.model[0]
                model_item.restored_file = restored_file
                restored_files.append(restored_file)
                logger.info(f"Single file restored file: {restored_file.get_path()}")
            else:
                assert os.path.isdir(restore_directory_or_file.get_path())
                restore_directory = restore_directory_or_file
                for idx, model_item in enumerate(self.model):
                    restored_file = self.get_restored_file_path(model_item.original_file, restore_directory.get_path())
                    model_item.restored_file = restored_file
                    restored_files.append(restored_file)
                    logger.info(f"Multiple file {idx} restored file: {restored_file.get_path()}")
            self.multiple_files_page.on_video_export_started(restored_files)

        item = self.model[self.get_next_queued_item_idx()]
        logger.info(f"Starting export for item: {item.original_file.get_path()} -> {item.restored_file.get_path()}")
        self._start_export(item.original_file, item.restored_file)

    def _start_export(self, source_file: Gio.File, restore_file: Gio.File):
        logger.info("=== STARTING EXPORT PROCESS ===")
        logger.info(f"Source file: {source_file.get_path()}")
        logger.info(f"Restore file: {restore_file.get_path()}")
        assert os.path.isfile(source_file.get_path())

        # Check for existing resume information for video splitting
        logger.info("STEP 1: Checking for resume info")
        logger.info(f"self.resume_info already set: {self.resume_info is not None}")
        logger.info(f"Video splitting enabled: {self._config.video_splitting_enabled}")

        # If we already have resume_info set (from user choosing resume), skip disk check
        if self.resume_info is None and self._config.video_splitting_enabled:
            logger.info("No existing resume_info, checking disk for resume info")
            logger.info(f"Checking for resume info for file: {source_file.get_path()}")
            resume_info = self._check_for_resume_info(source_file.get_path())
            logger.info(f"Resume info result: {resume_info}")
            if resume_info:
                logger.info("SUCCESS: Found resume info on disk, showing resume dialog")
                # Ask user if they want to resume
                self._show_resume_dialog(source_file, restore_file, resume_info)
                logger.info("Resume dialog shown, returning from _start_export")
                return
            else:
                logger.info("FAILURE: No resume info found on disk, proceeding with normal export")
        elif self.resume_info is not None:
            logger.info("SUCCESS: Using existing resume_info from user choice")
        else:
            logger.info("FAILURE: Video splitting not enabled, skipping resume check")

        logger.info("STEP 2: Starting export")
        logger.info(f"self.resume_info exists: {self.resume_info is not None}")
        logger.info("Showing video export started")
        self.show_video_export_started(restore_file)

        # Check if we should use video splitting or normal processing
        use_video_splitting = self._should_use_video_splitting(source_file.get_path())
        logger.info(f"Using video splitting: {use_video_splitting}")

        if use_video_splitting or self.resume_info is not None:
            logger.info("Starting video splitting processing")
            self._start_video_splitting_export(source_file, restore_file)
        else:
            logger.info("Starting normal processing (no splitting)")
            self._start_normal_export(source_file, restore_file)

    def _should_use_video_splitting(self, source_file_path: str) -> bool:
        """Determine if video splitting should be used for this video."""
        if not self._config.video_splitting_enabled:
            return False

        try:
            # Get video duration
            video_metadata = video_utils.get_video_meta_data(source_file_path)
            video_duration = video_metadata.duration
            part_duration_seconds = self._config.video_part_duration * 60  # Convert minutes to seconds

            logger.info(f"Video duration: {video_duration}s, Part duration: {part_duration_seconds}s")

            # If video is shorter than or equal to part duration, don't split
            if video_duration <= part_duration_seconds:
                logger.info("Video duration <= part duration, disabling splitting for this video")
                return False

            # If only 1 part would be created, don't use splitting (performance optimization)
            import math
            num_parts = math.ceil(video_duration / part_duration_seconds)
            if num_parts <= 1:
                logger.info("Only 1 part would be created, using normal processing for better performance")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error checking video duration, falling back to normal processing: {e}")
            return False

    def _start_normal_export(self, source_file: Gio.File, restore_file: Gio.File):
        """Start export using normal (non-splitting) processing logic."""
        def run_export():
            source_file_path = source_file.get_path()
            restore_file_path = restore_file.get_path()

            success = True

            # Normal processing logic (original)
            frame_restorer_options = FrameRestorerOptions(self._config.mosaic_restoration_model, self._config.mosaic_detection_model, video_utils.get_video_meta_data(source_file_path), self._config.device, self._config.max_clip_duration, False, False)
            video_metadata = frame_restorer_options.video_metadata
            frame_restorer_provider = FRAME_RESTORER_PROVIDER
            frame_restorer_provider.init(frame_restorer_options)
            frame_restorer = frame_restorer_provider.get()

            progress_update_step_size = 100
            temp_dir = self._config.temp_directory
            video_tmp_file_output_path = os.path.join(temp_dir, f"{os.path.basename(os.path.splitext(restore_file_path)[0])}.tmp{os.path.splitext(restore_file_path)[1]}")

            if self.resume_info:
                start_ns = self.resume_info.get_resume_timestamp_ns()
                start_frame_num = self.resume_info.frame_num
                logger.info(f"Resume requested: Starting FrameRestorer at timestamp {start_ns}ns")
            else:
                start_ns = 0
                start_frame_num = 0
                self.video_writer = video_utils.VideoWriter(
                    video_tmp_file_output_path, video_metadata.video_width,
                    video_metadata.video_height, video_metadata.video_fps_exact,
                    self._config.export_codec, time_base=video_metadata.time_base,
                    crf=self._config.export_crf, custom_encoder_options=self._config.custom_ffmpeg_encoder_options)
                self.progress_calculator = export_utils.ProgressCalculator(video_metadata)

            try:
                frame_restorer.start(start_ns=start_ns)

                duration_start = time.time()
                for frame_num, elem in enumerate(frame_restorer, start=start_frame_num):
                    if self.stop_requested:
                        success = False
                        logger.warning("Stop requested: Stopping FrameRestorer")
                        break
                    if elem is None:
                        success = False
                        logger.error("Error on export: frame restorer stopped prematurely")
                        break

                    (restored_frame, restored_frame_pts) = elem
                    if self.resume_info:
                        if restored_frame_pts <= self.resume_info.frame_pts:
                            logging.debug("Received frame earlier than resume position, skipping frame...")
                            continue
                        else:
                            logger.debug("Received first frame after resume position, successful resume.")
                            self.resume_info = None
                            # Ensure model item state is set to PROCESSING before emitting resume signal
                            GLib.idle_add(lambda: self._ensure_resume_state_transition())
                            GLib.idle_add(lambda: self.emit('video-export-resumed'))
                    self.video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)

                    duration_end = time.time()
                    duration = duration_end - duration_start
                    duration_start = duration_end
                    self.progress_calculator.update(duration)
                    if frame_num % progress_update_step_size == 0:
                        GLib.idle_add(lambda: self.emit('video-export-progress', self.progress_calculator.get_progress()))

                    if self.pause_requested:
                        logger.info("Pause requested: Pausing FrameRestorer")
                        self.resume_info = ResumeInformation(restored_frame_pts, video_metadata.time_base, frame_num)
                        break

            except Exception as e:
                success = False
                err_msg = "".join(traceback.format_exception_only(e))
                GLib.idle_add(lambda: self.emit('video-export-failed', err_msg))
            finally:
                if not self.pause_requested and self.video_writer is not None:
                    self.video_writer.release()
                frame_restorer.stop()

            # Handle success/failure for non-splitting case
            if self.pause_requested:
                GLib.idle_add(lambda: self.emit('video-export-paused'))
            elif success:
                audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, restore_file_path)
                def on_success():
                    progress = self.progress_calculator.get_progress()
                    progress.complete()
                    self.emit('video-export-progress', progress)
                    self.emit('video-export-finished')
                GLib.idle_add(on_success)
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)
                if self.stop_requested:
                    GLib.idle_add(lambda: self.emit('video-export-stopped'))

        exporter_thread = threading.Thread(target=run_export, daemon=True)
        exporter_thread.start()

    def _start_video_splitting_export(self, source_file: Gio.File, restore_file: Gio.File):
        """Start export using video splitting logic."""
        def run_export():
            source_file_path = source_file.get_path()
            restore_file_path = restore_file.get_path()

            success = True

            # Double-check if we should actually use video splitting (in case of resume)
            should_use_splitting = self._should_use_video_splitting(source_file_path)
            if not should_use_splitting and self.resume_info is None:
                logger.info("Video splitting not needed for this video, switching to normal processing")
                # Use normal processing instead
                self._start_normal_export(source_file, restore_file)
                return
            elif not should_use_splitting and self.resume_info is not None:
                logger.info("Video splitting not needed but resume info exists, keeping resume info and using normal processing")
                # Keep resume info since we need it for normal processing resume
                self._start_normal_export(source_file, restore_file)
                return

            # Check if video splitting is enabled
            if self._config.video_splitting_enabled:
                logger.info("Video splitting enabled - starting split processing")
                logger.debug(f"Initial pause_requested state: {self.pause_requested}")
                # Use video splitting logic for crash recovery
                import tempfile
                import shutil
                # Use a persistent directory name based on input file path to allow resume across sessions
                import hashlib
                file_hash = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
                parts_dir = os.path.join(self._config.temp_directory, f"lada_parts_{file_hash}")
                os.makedirs(parts_dir, exist_ok=True)

                # Save initial resume info for part 0
                resume_info_initial = export_utils.ResumeInformation(0, Fraction(1, 30), 0, 0.0)
                self._save_resume_info(source_file_path, resume_info_initial)
                logger.debug("Saved initial resume info for part 0")

                success = False
                try:
                    # Split video into parts
                    part_duration_seconds = self._config.video_part_duration * 60  # Convert minutes to seconds
                    part_files = video_utils.split_video_by_duration(source_file_path, os.path.join(parts_dir, "part_%03d.mp4"), part_duration_seconds)

                    # Save initial resume info for part 0
                    resume_info_initial = export_utils.ResumeInformation(0, Fraction(1, 30), 0, 0.0)
                    self._save_resume_info(source_file_path, resume_info_initial)
                    logger.debug("Saved initial resume info for part 0")

                    # Process each part sequentially
                    processed_parts = []
                    total_parts = len(part_files)

                    # Initialize overall progress tracking first
                    overall_progress_calculator = export_utils.ProgressCalculator(video_utils.get_video_meta_data(source_file_path))
                    total_processing_time = 0.0

                    # Handle resume for video splitting
                    processed_parts = []  # Initialize list for processed parts
                    start_part_idx = 0
                    total_time_from_previous_parts = 0.0

                    if self.resume_info:
                        start_part_idx = self.resume_info.frame_num  # Using frame_num to store part index
                        # Get total time from previous parts from resume info
                        total_time_from_previous_parts = self.resume_info.total_processing_time_s
                        logger.info(f"Resuming video splitting from part {start_part_idx + 1}")
                        logger.info(f"Resume info: part_idx={start_part_idx}, total_time_from_previous_parts={total_time_from_previous_parts}")
                        self.resume_info = None  # Clear resume info
                        logger.info("Cleared resume_info and will emit video-export-resumed signal")
                        # Don't emit resume signal here - it will be emitted after progress calculation

                        # Check for already processed parts from previous runs
                        for i in range(total_parts):
                            expected_output = os.path.join(parts_dir, f"processed_part_{i+1:03d}.mp4")
                            if os.path.exists(expected_output):
                                processed_parts.append(expected_output)
                                logger.info(f"Found existing processed part: {expected_output}")
                            else:
                                logger.warning(f"Expected processed part not found: {expected_output}")

                    # When resuming, account for already completed parts in progress calculation
                    if start_part_idx > 0:
                        # Calculate progress based on actual part durations
                        total_video_duration = overall_progress_calculator.video_metadata.duration
                        completed_duration = 0.0

                        # Sum durations of completed parts
                        for i in range(start_part_idx):
                            if i < len(part_files):
                                part_metadata = video_utils.get_video_meta_data(part_files[i])
                                completed_duration += part_metadata.duration

                        completed_parts_fraction = completed_duration / total_video_duration if total_video_duration > 0 else 0
                        overall_progress_calculator.frames_done = int(completed_parts_fraction * overall_progress_calculator.video_metadata.frames_count)
                        overall_progress_calculator.time_done_s = total_time_from_previous_parts
                        logger.info(f"Resuming with {completed_parts_fraction:.1%} progress already completed (completed {completed_duration:.1f}s of {total_video_duration:.1f}s, total time spent: {total_time_from_previous_parts:.1f}s)")

                        # Emit initial progress to show resumed state in GUI
                        initial_progress = ExportItemDataProgress()
                        initial_progress.fraction = completed_parts_fraction
                        initial_progress.frames_done = int(completed_parts_fraction * overall_progress_calculator.video_metadata.frames_count)
                        initial_progress.frames_remaining = overall_progress_calculator.video_metadata.frames_count - initial_progress.frames_done
                        initial_progress.time_done_s = total_time_from_previous_parts
                        initial_progress.speed_fps = 0  # Will be updated during processing
                        initial_progress.time_remaining_s = 0  # Will be updated during processing
                        initial_progress.enough_datapoints = False
                        GLib.idle_add(lambda: self.emit('video-export-progress', initial_progress))
                        # Emit resume signal after progress is set to ensure GUI updates correctly
                        GLib.idle_add(lambda: self.emit('video-export-resumed'))

                    # Process remaining parts (skip parts that were already processed) - LIGHTWEIGHT LOOP
                    # Save resume info for the starting part
                    if start_part_idx < total_parts:
                        resume_info_start = export_utils.ResumeInformation(0, Fraction(1, 30), start_part_idx, total_time_from_previous_parts if start_part_idx > 0 else 0.0)
                        self._save_resume_info(source_file_path, resume_info_start)
                        logger.debug(f"Saved resume info for starting part {start_part_idx}: total_time={total_time_from_previous_parts if start_part_idx > 0 else 0.0}")


                    for part_idx in range(len(processed_parts), total_parts):
                        part_path = part_files[part_idx]
                        logger.debug(f"Processing video part {part_idx + 1}/{total_parts}: {os.path.basename(part_path)}")
                        if self.stop_requested:
                            logger.info("Stop requested during video splitting - breaking out of part loop")
                            success = False
                            break
                        if self.pause_requested:
                            logger.info(f"Pause requested during video splitting at part {part_idx + 1}")
                            # Save resume information for video splitting - use part index as frame number
                            self.resume_info = export_utils.ResumeInformation(0, Fraction(1, 30), part_idx, total_processing_time)
                            self._save_resume_info(source_file_path, self.resume_info)
                            logger.info(f"Set resume_info for video splitting: part_idx={part_idx}, total_time_so_far={total_processing_time}")
                            success = False
                            break

                        part_output_path = os.path.join(parts_dir, f"processed_{os.path.basename(part_path)}")

                        # Process this part with lightweight progress tracking
                        # Use total_time_from_previous_parts for resume, otherwise use accumulated total_processing_time
                        time_from_previous_parts = total_time_from_previous_parts if start_part_idx > 0 and part_idx == start_part_idx else total_processing_time
                        part_processing_time = self._process_single_video_part_with_progress(
                            part_path, part_output_path, self._config, part_idx, total_parts, overall_progress_calculator, time_from_previous_parts, part_files, source_file_path
                        )
                        if part_processing_time is not None:
                            total_processing_time += part_processing_time
                            processed_parts.append(part_output_path)
                            # Save resume info for next part after successful processing
                            next_part_idx = part_idx + 1
                            if next_part_idx < total_parts:
                                resume_info_next = export_utils.ResumeInformation(0, Fraction(1, 30), next_part_idx, total_processing_time)
                                self._save_resume_info(source_file_path, resume_info_next)
                                logger.debug(f"Saved resume info for next part: part_idx={next_part_idx}, total_time={total_processing_time}")
                        else:
                            logger.error(f"Failed to process part {part_idx + 1}")
                            success = False
                            break

                    # Merge processed parts if all succeeded
                    logger.info(f"Processed parts: {len(processed_parts)}, Total parts: {len(part_files)}, Stop requested: {self.stop_requested}")
                    if len(processed_parts) == len(part_files) and not self.stop_requested:
                        logger.info("Merging processed parts...")
                        logger.info(f"Parts to merge: {processed_parts}")
                        video_utils.merge_video_parts(processed_parts, restore_file_path)
                        success = True
                        logger.info("Video splitting completed successfully")
                    else:
                        logger.info("Not all parts processed successfully or stop was requested")
                        success = False

                except Exception as e:
                    logger.error(f"Error during video splitting processing: {e}")
                    success = False

                # Handle success/failure for video splitting
                if self.pause_requested:
                    logger.info(f"Video splitting paused - resume_info set: {self.resume_info is not None}")
                    if self.resume_info:
                        logger.info(f"Resume info details: frame_num={self.resume_info.frame_num}")
                    GLib.idle_add(lambda: self.emit('video-export-paused'))
                elif success and not self.stop_requested:
                    GLib.idle_add(lambda: self.emit('video-export-finished'))
                    # Cleanup after successful completion
                    try:
                        shutil.rmtree(parts_dir)
                        logger.info("Cleaned up temporary parts directory after successful completion")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary directory: {cleanup_error}")
                    # Clear resume info since export completed
                    self._clear_resume_info(source_file_path)
                elif self.stop_requested:
                    GLib.idle_add(lambda: self.emit('video-export-stopped'))
                    # Cleanup after cancellation
                    try:
                        shutil.rmtree(parts_dir)
                        logger.info("Cleaned up temporary parts directory after cancellation")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary directory: {cleanup_error}")
                else:
                    error_msg = str(e) if 'e' in locals() else "Video splitting failed"
                    GLib.idle_add(lambda: self.emit('video-export-failed', error_msg))
                    # Cleanup after failure
                    try:
                        shutil.rmtree(parts_dir)
                        logger.info("Cleaned up temporary parts directory after failure")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary directory: {cleanup_error}")

            else:
                logger.error("Video splitting logic should not reach this point")
                success = False

        exporter_thread = threading.Thread(target=run_export, daemon=True)
        exporter_thread.start()

    def _process_single_video_part_with_progress(self, part_path: str, output_path: str, config: Config, part_idx: int, total_parts: int, overall_progress_calculator: export_utils.ProgressCalculator, total_processing_time_so_far: float, part_files: list[str], source_file_path: str) -> float | None:
        """Process a single video part for splitting functionality - LIGHTWEIGHT VERSION for maximum speed."""
        try:
            # Use the same lightweight processing as normal export for maximum speed
            frame_restorer_options = FrameRestorerOptions(
                config.mosaic_restoration_model,
                config.mosaic_detection_model,
                video_utils.get_video_meta_data(part_path),
                config.device,
                config.max_clip_duration,
                False,
                False
            )
            video_metadata = frame_restorer_options.video_metadata
            frame_restorer_provider = FRAME_RESTORER_PROVIDER
            frame_restorer_provider.init(frame_restorer_options)
            frame_restorer = frame_restorer_provider.get()

            success = True
            temp_dir = config.temp_directory
            video_tmp_file_output_path = os.path.join(temp_dir, f"{os.path.basename(os.path.splitext(output_path)[0])}.tmp{os.path.splitext(output_path)[1]}")

            video_writer = video_utils.VideoWriter(
                video_tmp_file_output_path, video_metadata.video_width,
                video_metadata.video_height, video_metadata.video_fps_exact,
                config.export_codec, time_base=video_metadata.time_base,
                crf=config.export_crf, custom_encoder_options=config.custom_ffmpeg_encoder_options
            )

            # Simple progress calculator for this part only
            progress_calculator = export_utils.ProgressCalculator(video_metadata)

            frame_restorer.start()

            part_start_time = time.time()
            duration_start = time.time()
            progress_update_step_size = 100  # Same as normal processing

            for frame_num, elem in enumerate(frame_restorer):
                if self.stop_requested:
                    success = False
                    logger.info("Stop requested during video part processing")
                    break
                if self.pause_requested:
                    success = False
                    logger.info("Pause requested during video part processing")
                    # Set resume info for video splitting - use current part index
                    self.resume_info = export_utils.ResumeInformation(0, Fraction(1, 30), part_idx, total_processing_time_so_far + part_start_time - time.time())
                    # Save resume info to disk for persistence across app restarts
                    self._save_resume_info(source_file_path, self.resume_info)
                    logger.info(f"Set resume_info during part processing: part_idx={part_idx}, total_time_so_far={self.resume_info.total_processing_time_s}")
                    break
                if elem is None:
                    success = False
                    break

                (restored_frame, restored_frame_pts) = elem
                video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)

                duration_end = time.time()
                duration = duration_end - duration_start
                duration_start = duration_end
                progress_calculator.update(duration)

                # LIGHTWEIGHT PROGRESS UPDATE - Simple calculation for speed
                if frame_num % progress_update_step_size == 0:
                    # Calculate simple overall progress without complex duration math
                    part_progress = progress_calculator.get_progress()
                    # Estimate overall progress: completed parts + current part progress
                    completed_parts_fraction = part_idx / total_parts
                    current_part_fraction = part_progress.fraction / total_parts
                    overall_fraction = completed_parts_fraction + current_part_fraction

                    overall_progress = ExportItemDataProgress()
                    overall_progress.fraction = min(overall_fraction, 1.0)
                    overall_progress.frames_done = int(overall_fraction * overall_progress_calculator.video_metadata.frames_count)
                    overall_progress.frames_remaining = overall_progress_calculator.video_metadata.frames_count - overall_progress.frames_done
                    overall_progress.time_done_s = total_processing_time_so_far + part_progress.time_done_s
                    overall_progress.speed_fps = part_progress.speed_fps
                    # For video splitting, calculate remaining time based on overall progress, not current part
                    if overall_progress.speed_fps > 0 and overall_progress.frames_remaining > 0:
                        overall_progress.time_remaining_s = overall_progress.frames_remaining / overall_progress.speed_fps
                    else:
                        overall_progress.time_remaining_s = part_progress.time_remaining_s  # Fallback
                    overall_progress.enough_datapoints = part_progress.enough_datapoints

                    GLib.idle_add(lambda: self.emit('video-export-progress', overall_progress))

            video_writer.release()
            frame_restorer.stop()
            part_end_time = time.time()
            part_processing_time = part_end_time - part_start_time

            if success:
                # Add audio to the processed part
                audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_path)

                # Final progress update for completed part
                completed_parts_fraction = (part_idx + 1) / total_parts
                overall_progress = ExportItemDataProgress()
                overall_progress.fraction = min(completed_parts_fraction, 1.0)
                overall_progress.frames_done = int(completed_parts_fraction * overall_progress_calculator.video_metadata.frames_count)
                overall_progress.frames_remaining = overall_progress_calculator.video_metadata.frames_count - overall_progress.frames_done
                overall_progress.time_done_s = total_processing_time_so_far + part_processing_time
                overall_progress.speed_fps = video_metadata.frames_count / part_processing_time if part_processing_time > 0 else 0
                overall_progress.time_remaining_s = 0  # Will be calculated for remaining parts
                overall_progress.enough_datapoints = True

                GLib.idle_add(lambda: self.emit('video-export-progress', overall_progress))

                return part_processing_time
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)
                return None

        except Exception as e:
            logger.error(f"Error processing video part {part_path}: {e}")
            return None

    def show_export_dialog(self, dismissed_callback):
        def on_dialog_result(dialog, result):
            try:
                if self.single_file:
                    selected = dialog.save_finish(result)
                else:
                    selected = dialog.select_folder_finish(result)
                if selected is not None:
                    self.emit("video-export-requested",selected)
            except GLib.Error as error:
                if error.code == 2: # "Dismissed by user"
                    dismissed_callback()
                    logger.debug("FileDialog cancelled: Dismissed by user")
                else:
                    logger.error(f"Error opening file: {error.message}")
                    raise error

        if self.single_file:
            file_dialog = Gtk.FileDialog()
            video_file_filter = Gtk.FileFilter()
            video_file_filter.add_mime_type("video/*")
            file_dialog.set_default_filter(video_file_filter)
            file_dialog.set_title(_("Save restored video file"))
            initial_restored_file = self.model[0].restored_file
            file_dialog.set_initial_folder(initial_restored_file.get_parent())
            file_dialog.set_initial_name(initial_restored_file.get_basename())
            file_dialog.save(callback=on_dialog_result)
        else:
            file_dialog = Gtk.FileDialog()
            file_dialog.set_title(_("Save restored video files"))
            first_original_file = self.model[0].original_file
            file_dialog.set_initial_folder(first_original_file.get_parent())
            file_dialog.select_folder(callback=on_dialog_result)

    def get_restored_file_path(self, original_file: Gio.File, output_dir: str) -> Gio.File:
        orig_file_name = os.path.splitext(original_file.get_basename())[0]
        restored_file_name = self._config.file_name_pattern.replace("{orig_file_name}", orig_file_name)
        return Gio.File.new_build_filenamev([output_dir, restored_file_name])

    def execute_post_export_action(self):
        action = self._config.post_export_action
        if action == PostExportAction.NONE:
            return
        elif action == PostExportAction.SHUTDOWN:
            logger.info("Post-export action: Shutting down PC - showing confirmation dialog")
            self.show_shutdown_confirmation_dialog()
            self.emit("shutdown-confirmation-requested")
        elif action == PostExportAction.CUSTOM_COMMAND:
            command = self._config.post_export_custom_command.strip()
            if command:
                logger.info(f"Post-export action: Executing custom command: {command}")
                import subprocess
                try:
                    subprocess.Popen(command, shell=True)
                except Exception as e:
                    logger.error(f"Failed to execute custom command '{command}': {e}")

    def show_shutdown_confirmation_dialog(self):
        dialog = Adw.AlertDialog(
            heading=_("Shutdown System"),
            body=_("Export has finished. The system will shutdown in 30 seconds."),
        )

        timeout_id = None

        def execute_shutdown():
            logger.info("Timeout reached - proceeding with automatic shutdown")
            try:
                shutdown_manager = ShutdownManager()
                shutdown_manager.shutdown()
                logger.info("Shutdown command executed successfully")
            except ShutdownError as e:
                logger.error(f"Failed to initiate shutdown: {e}")
                error_dialog = Adw.AlertDialog(
                    heading=_("Shutdown Failed"),
                    body=_("Failed to initiate system shutdown. Please check system permissions."),
                )
                error_dialog.add_response("ok", _("Okay"))
                error_dialog.choose(self, None, lambda _dialog, task: _dialog.choose_finish(task))

        def on_response_selected(_dialog, task: Gio.Task):
            nonlocal timeout_id
            response = _dialog.choose_finish(task)
            if timeout_id:
                GLib.source_remove(timeout_id)
            if response == "shutdown":
                logger.info("User confirmed shutdown - proceeding with system shutdown")
                execute_shutdown()
            else:
                logger.info("User cancelled shutdown")

        timeout_id = GLib.timeout_add_seconds(30, lambda: execute_shutdown())

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("shutdown", _("Shutdown now"))
        dialog.set_response_appearance("shutdown", Adw.ResponseAppearance.DESTRUCTIVE)

        dialog.choose(self, None, on_response_selected)

    def _check_for_resume_info(self, source_file_path: str) -> ResumeInformation | None:
        """Check if there's existing resume information for video splitting."""
        import hashlib
        import json

        logger.info(f"Checking resume info for path: {source_file_path}")
        file_hash = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        logger.info(f"File hash: {file_hash}")
        parts_dir = os.path.join(self._config.temp_directory, f"lada_parts_{file_hash}")
        logger.info(f"Parts dir: {parts_dir}")
        resume_file = os.path.join(parts_dir, "resume_info.json")
        logger.info(f"Resume file path: {resume_file}")
        logger.info(f"Resume file exists: {os.path.exists(resume_file)}")

        if os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                logger.info(f"Resume data loaded: {resume_data}")
                resume_info = ResumeInformation(
                    resume_data['frame_pts'],
                    Fraction(resume_data['time_base_num'], resume_data['time_base_den']),
                    resume_data['frame_num'],
                    resume_data.get('total_processing_time_s', 0.0)  # New field with default
                )
                logger.info(f"Resume info created: frame_num={resume_info.frame_num}, total_time={resume_info.total_processing_time_s}")
                return resume_info
            except Exception as e:
                logger.warning(f"Failed to load resume info: {e}")
                return None
        logger.info("Resume file does not exist")
        return None

    def _save_resume_info(self, source_file_path: str, resume_info: ResumeInformation):
        """Save resume information to disk for video splitting."""
        import hashlib
        import json

        file_hash = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        parts_dir = os.path.join(self._config.temp_directory, f"lada_parts_{file_hash}")
        os.makedirs(parts_dir, exist_ok=True)

        resume_file = os.path.join(parts_dir, "resume_info.json")
        resume_data = {
            'frame_pts': resume_info.frame_pts,
            'time_base_num': resume_info.time_base.numerator,
            'time_base_den': resume_info.time_base.denominator,
            'frame_num': resume_info.frame_num,
            'total_processing_time_s': resume_info.total_processing_time_s
        }

        try:
            with open(resume_file, 'w') as f:
                json.dump(resume_data, f)
        except Exception as e:
            logger.warning(f"Failed to save resume info: {e}")

    def _show_resume_dialog(self, source_file: Gio.File, restore_file: Gio.File, resume_info: ResumeInformation):
        """Show dialog asking user if they want to resume or start fresh."""
        logger.info(f"Showing resume dialog for file: {source_file.get_path()}")
        logger.info(f"Resume info: frame_num={resume_info.frame_num}, frame_pts={resume_info.frame_pts}")

        dialog = Adw.AlertDialog(
            heading=_("Resume Previous Export"),
            body=_("A previous export was interrupted. Would you like to resume from where it left off?"),
        )

        def on_resume_response(dialog, task):
            try:
                response = dialog.choose_finish(task)
                logger.info(f"Resume dialog response: {response}")
                if response == "resume":
                    # Set resume info and start export
                    logger.info("User chose to resume - setting resume_info and starting export")
                    self.resume_info = resume_info
                    self.show_video_export_started(restore_file)
                    self._start_export(source_file, restore_file)
                elif response == "fresh":
                    # Clear any existing resume info and delete parts directory, then start fresh
                    logger.info("User chose to start fresh - clearing resume info and deleting parts directory")
                    self._clear_resume_info_and_parts(source_file.get_path())
                    self.show_video_export_started(restore_file)
                    self._start_export(source_file, restore_file)
                else:
                    logger.info(f"User cancelled resume dialog (response: {response})")
            except Exception as e:
                logger.error(f"Error handling resume dialog response: {e}")

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("fresh", _("Start Fresh"))
        dialog.add_response("resume", _("Resume"))
        dialog.set_response_appearance("resume", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_response_appearance("fresh", Adw.ResponseAppearance.DESTRUCTIVE)

        dialog.choose(self, None, on_resume_response)

    def _clear_resume_info(self, source_file_path: str):
        """Clear any existing resume information."""
        import hashlib

        file_hash = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        parts_dir = os.path.join(self._config.temp_directory, f"lada_parts_{file_hash}")
        resume_file = os.path.join(parts_dir, "resume_info.json")

        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
            except Exception as e:
                logger.warning(f"Failed to clear resume info: {e}")

    def _clear_resume_info_and_parts(self, source_file_path: str):
        """Clear resume information and delete the entire parts directory."""
        import hashlib
        import shutil

        file_hash = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        parts_dir = os.path.join(self._config.temp_directory, f"lada_parts_{file_hash}")

        # Remove resume file first
        resume_file = os.path.join(parts_dir, "resume_info.json")
        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
                logger.info(f"Removed resume info file: {resume_file}")
            except Exception as e:
                logger.warning(f"Failed to remove resume info file: {e}")

        # Remove entire parts directory
        if os.path.exists(parts_dir):
            try:
                shutil.rmtree(parts_dir)
                logger.info(f"Removed parts directory: {parts_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove parts directory: {e}")

    def close(self):
        self.stop_requested = True
