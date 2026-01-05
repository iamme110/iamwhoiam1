# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
import os
from gi.repository import Adw, Gtk, Gio, GObject, GLib
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.fileselection.url_input_dialog import UrlInputDialog
from lada.gui.shortcuts import ShortcutsManager
from lada.utils import video_utils
from lada.gui.config.config import Config
from gettext import gettext as _

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(str(here / 'file_selection_view.ui')))
class FileSelectionView(Gtk.Widget):
    __gtype_name__ = 'FileSelectionView'

    button_open_file: Gtk.Button = Gtk.Template.Child()
    button_watch_url: Gtk.Button = Gtk.Template.Child()
    status_page: Adw.StatusPage = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._shortcuts_manager: ShortcutsManager | None = None
        self._current_downloaded_file: str | None = None
        self._estimated_total_mb: float | None = None
        self._config = None
        
        # Download completion tracking
        self._download_completed = False
        self._completed_file_path: str | None = None
        self._download_temp_directory: str | None = None
        
        # File cleanup tracking - only rename when GUI closes
        self._video_file_for_cleanup: str | None = None
        self._watch_now_processing: bool = False
        self._autostart_playback: bool = False  # New: autostart playback setting
        
        # Process tracking for cancellation
        self._current_process = None
        self._process_holder = None  # Store process holder as instance variable
        self._download_thread = None
        self._cancelled = False  # Flag to indicate cancellation was requested
        
        self._setup_gui_close_cleanup()

        drop_target = utils.create_video_files_drop_target(lambda files: self.emit("files-selected", files))
        self.add_controller(drop_target)

        logo_image = Gtk.Image.new_from_resource("/io/github/ladaapp/lada/icons/128x128/lada-logo-gray.png")
        self.status_page.set_paintable(logo_image.get_paintable())

    @Gtk.Template.Callback()
    def button_open_file_callback(self, button_clicked):
        self.button_open_file.set_sensitive(False)
        callback = lambda files: self.emit("files-selected", files)
        dismissed_callback = lambda *args: self.button_open_file.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    @Gtk.Template.Callback()
    def button_watch_url_callback(self, button_clicked):
        self.button_watch_url.set_sensitive(False)
        self._show_url_input_dialog()

    def _show_url_input_dialog(self):
        """Show dialog for URL input"""
        try:
            dialog = UrlInputDialog()
            dialog.connect("url-confirmed", self._on_url_confirmed)
            dialog.connect("dialog-dismissed", lambda *args: self.button_watch_url.set_sensitive(True))
            
            # Present the dialog as a modal window
            dialog.present()
            
            # Try to focus the URL entry immediately
            GLib.idle_add(lambda: self._focus_dialog_entry(dialog))
                
        except Exception as e:
            logger.error(f"Failed to create URL input dialog: {e}")
            self._show_download_error_dialog(f"Failed to open URL dialog: {str(e)}")
            self.button_watch_url.set_sensitive(True)
    
    def _focus_dialog_entry(self, dialog):
        """Focus the URL entry in the dialog"""
        try:
            if hasattr(dialog, '_url_entry') and dialog._url_entry:
                dialog._url_entry.grab_focus()
        except Exception as e:
            pass  # Silent error handling

    def _on_url_confirmed(self, dialog, url: str):
        """Handle confirmed URL"""
        self.button_watch_url.set_sensitive(True)
        dialog.close()
        
        # Show progress dialog and start progressive download
        self._show_progress_dialog()
        
        # Start download in background thread
        import threading
        self._download_thread = threading.Thread(target=self._download_video_progressive, args=(url,))
        self._download_thread.daemon = True
        self._download_thread.start()

    def _show_progress_dialog(self):
        """Show download progress dialog"""
        # Reset download completion tracking for new download
        self._download_completed = False
        self._completed_file_path = None
        
        # Create very compact custom dialog without progress bar
        self.progress_dialog = Adw.ApplicationWindow()
        self.progress_dialog.set_title("Downloading Video")
        self.progress_dialog.set_default_size(320, 140)
        self.progress_dialog.set_resizable(False)
        
        # Create main content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        
        # Status label
        self._progress_label = Gtk.Label()
        self._progress_label.set_text("Preparing download...")
        self._progress_label.set_halign(Gtk.Align.START)
        main_box.append(self._progress_label)
        
        # Size info label
        self._size_label = Gtk.Label()
        self._size_label.set_text("0 MB / 0 MB (0%)")
        self._size_label.set_halign(Gtk.Align.START)
        self._size_label.add_css_class("dim-label")
        main_box.append(self._size_label)
        
        # Autostart checkbox
        autostart_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        autostart_box.set_halign(Gtk.Align.START)
        
        self._autostart_checkbox = Gtk.CheckButton.new_with_label("Autostart playback")
        self._autostart_checkbox.set_active(False)
        autostart_box.append(self._autostart_checkbox)
        main_box.append(autostart_box)
        
        # Button box
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        button_box.set_halign(Gtk.Align.END)
        
        # Watch Now button (initially disabled)
        self._watch_now_button = Gtk.Button.new_with_label("Watch Now")
        self._watch_now_button.add_css_class("suggested-action")
        self._watch_now_button.set_sensitive(False)
        self._watch_now_button.connect("clicked", self._on_watch_now_clicked)
        button_box.append(self._watch_now_button)
        
        # Cancel button
        cancel_button = Gtk.Button.new_with_label("Cancel")
        cancel_button.connect("clicked", self._on_progress_cancel_clicked)
        button_box.append(cancel_button)
        
        main_box.append(button_box)
        
        self.progress_dialog.set_content(main_box)
        self.progress_dialog.present()

    def _on_progress_dialog_response(self, dialog, response_id: str):
        """Handle progress dialog responses"""
        if response_id == "cancel":
            self._cancel_download()
            self.progress_dialog = None
    
    def _on_progress_cancel_clicked(self, button):
        """Handle progress dialog cancel button click"""
        self._cancel_download()
        self._close_progress_dialog()
    
    def _on_watch_now_clicked(self, button):
        """Handle watch now button click"""
        # Prevent double-click issues by checking if already processing
        if hasattr(self, '_watch_now_processing') and self._watch_now_processing:
            logger.info("Watch Now already processing, ignoring duplicate click")
            return
        
        try:
            # Mark as processing to prevent double-clicks
            self._watch_now_processing = True
            
            # Close the progress dialog immediately when "Watch Now" is clicked
            self._close_progress_dialog()
            
            # Use the current downloaded file path directly since we know it exists
            if hasattr(self, '_current_downloaded_file') and self._current_downloaded_file:
                current_file = pathlib.Path(self._current_downloaded_file)
                logger.info(f"Watch Now: Using current file: {current_file}")
                
                if current_file.exists():
                    file_size = current_file.stat().st_size
                    logger.info(f"Current file exists, size: {file_size/1024/1024:.1f} MB")
                    
                    if file_size >= 20 * 1024 * 1024:  # 20MB threshold
                        gio_file = Gio.File.new_for_path(str(current_file))
                        logger.info(f"Emitting files-selected signal with: {current_file}")
                        self.emit("files-selected", [gio_file])
                        
                        # Store the file path for cleanup when GUI closes (NOT during playback)
                        self._video_file_for_cleanup = str(current_file)
                        
                        logger.info(f"Successfully started playback of: {current_file}")
                        return
                    else:
                        logger.info(f"File too small for playback: {file_size/1024/1024:.1f} MB < 20 MB")
                        # Don't show error - just let download continue
                else:
                    logger.info(f"Current file doesn't exist: {current_file}")
                    # Try to find the file in the temp directory
                    if hasattr(self, '_download_temp_directory') and self._download_temp_directory:
                        download_dir = pathlib.Path(self._download_temp_directory)
                        logger.info(f"Searching in temp directory: {download_dir}")
                        
                        # Look for stream files
                        for stream_file in download_dir.glob("stream.*"):
                            if stream_file.stat().st_size >= 20 * 1024 * 1024:
                                gio_file = Gio.File.new_for_path(str(stream_file))
                                logger.info(f"About to emit files-selected signal with: {stream_file}")
                                self.emit("files-selected", [gio_file])
                                
                                # Store the file path for cleanup when GUI closes (NOT during playback)
                                self._video_file_for_cleanup = str(stream_file)
                                
                                logger.info(f"Successfully emitted files-selected signal for: {stream_file}")
                                return
                        
                        # Look for any video files
                        for video_ext in ['*.mp4', '*.webm', '*.mkv', '*.avi', '*.mov', '*.m4v']:
                            for video_file in download_dir.glob(video_ext):
                                if video_file.stat().st_size >= 20 * 1024 * 1024:
                                    gio_file = Gio.File.new_for_path(str(video_file))
                                    self.emit("files-selected", [gio_file])
                                    
                                    # Store the file path for cleanup when GUI closes (NOT during playback)
                                    self._video_file_for_cleanup = str(video_file)
                                    
                                    logger.info(f"Found and playing video file: {video_file}")
                                    return
            else:
                logger.info("No downloaded file path available")
                        
        except Exception as e:
            # Don't show error dialog - just log and continue
            logger.error(f"Error starting to watch: {e}")
            # Don't call _show_download_error_dialog - let the download continue in background
        finally:
            # Reset processing flag after a short delay to ensure the signal is processed
            def reset_processing():
                self._watch_now_processing = False
            GLib.timeout_add(100, reset_processing)  # 100ms delay
    
    def _setup_gui_close_cleanup(self):
        """Set up cleanup (file renaming disabled to prevent GUI hang)"""
        # File renaming disabled to prevent GUI hanging when files are locked
        # Users can manually rename .part files if needed
        pass
    
    def _cleanup_downloaded_files_on_close(self):
        """Cleanup method - file renaming disabled to prevent GUI hang"""
        # File renaming functionality has been disabled
        # This prevents the GUI from hanging when trying to rename locked files
        # Users can manually rename .part files if needed
        pass
    
    def connect_gui_close_cleanup(self, window):
        """Connect the cleanup method to the window's close signal"""
        try:
            # Connect to window destroy signal
            if hasattr(window, 'destroy'):
                window.connect('destroy', lambda *args: self._cleanup_downloaded_files_on_close())
            # Also try to connect to application quit signal
            app = window.get_application()
            if app:
                app.connect('shutdown', lambda *args: self._cleanup_downloaded_files_on_close())
        except Exception as e:
            logger.debug(f"Could not connect GUI close cleanup: {e}")

    def _close_progress_dialog(self):
        """Close and clean up the progress dialog"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            try:
                self.progress_dialog.close()
            except:
                pass
            self.progress_dialog = None

    def _cancel_download(self):
        """Cancel the current download"""
        import os
        import signal
        import time
        
        cancel_time = time.time()
        logger.info(f"Cancelling download at {cancel_time}...")
        logger.info(f"Current process: {getattr(self, '_current_process', 'None')}")
        logger.info(f"Process holder: {getattr(self, '_process_holder', 'None')}")
        
        # Store cancel time for debugging
        self._cancel_time = cancel_time
        
        # Try to find and cancel the process from multiple sources
        process_to_cancel = None
        if hasattr(self, '_current_process') and self._current_process:
            process_to_cancel = self._current_process
            logger.info(f"Using tracked process: {process_to_cancel}")
        elif hasattr(self, '_process_holder') and self._process_holder and len(self._process_holder) > 0 and self._process_holder[0]:
            process_to_cancel = self._process_holder[0]
            logger.info(f"Using process holder: {process_to_cancel}")
        else:
            logger.warning("No process reference found, will use system-wide kill")
        
        # Terminate the yt-dlp process if it exists
        if process_to_cancel:
            try:
                process = process_to_cancel
                logger.info(f"Process object: {process}, PID: {getattr(process, 'pid', 'No PID')}")
                
                # Check if process is still running
                poll_result = process.poll()
                logger.info(f"Process poll result: {poll_result}")
                
                if poll_result is None:  # Process is still running
                    logger.info(f"Terminating yt-dlp process PID: {process.pid}")
                    
                    # Set global cancellation flag IMMEDIATELY to prevent error dialog
                    try:
                        from lada.gui.utils import _download_cancelled
                        _download_cancelled = True
                        logger.info("Set global download cancellation flag IMMEDIATELY")
                    except ImportError:
                        logger.warning("Could not import global cancellation flag")
                    
                    # First, immediately try to kill all yt-dlp processes system-wide
                    logger.info("First, killing all yt-dlp processes system-wide...")
                    self._kill_remaining_yt_dlp_processes()
                    
                    # Then close stdout to stop progress monitoring immediately
                    try:
                        if process.stdout:
                            logger.info("Closing process stdout to stop progress monitoring")
                            process.stdout.close()
                    except Exception as e:
                        logger.warning(f"Error closing stdout: {e}")
                    
                    # Try graceful termination of our tracked process
                    try:
                        if os.name == 'nt':  # Windows
                            logger.info("Using Windows process.terminate()")
                            process.terminate()
                        else:  # Unix-like systems
                            logger.info(f"Using Unix kill with SIGTERM for PID {process.pid}")
                            os.kill(process.pid, signal.SIGTERM)
                    except Exception as e:
                        logger.error(f"Error during graceful termination: {e}")
                    
                    # Wait a moment for graceful termination
                    try:
                        logger.info("Waiting for graceful termination...")
                        process.wait(timeout=2)
                        logger.info("Process terminated gracefully")
                    except Exception as e:
                        logger.warning(f"Graceful termination failed: {e}")
                        # If graceful termination fails, force kill
                        try:
                            if os.name == 'nt':  # Windows
                                logger.info("Using Windows process.kill()")
                                process.kill()
                            else:  # Unix-like systems
                                logger.info(f"Using Unix kill with SIGKILL for PID {process.pid}")
                                os.kill(process.pid, signal.SIGKILL)
                            process.wait(timeout=1)
                            logger.info("Process killed forcefully")
                        except Exception as e:
                            logger.error(f"Error during forced kill: {e}")
                else:
                    logger.info(f"Process already finished with return code: {poll_result}")
                
            except Exception as e:
                logger.error(f"Error terminating download process: {e}")
        else:
            logger.warning("No current process to cancel")
        
        # Clear the process reference
        self._current_process = None
        
        # Stop progress monitoring thread
        self._stop_progress_monitoring = True
        
        # Reset download thread
        if hasattr(self, '_download_thread') and self._download_thread:
            # Daemon threads will be terminated when the main thread exits
            self._download_thread = None
        
        # Reset progress thread tracking
        if hasattr(self, '_progress_thread') and self._progress_thread:
            self._progress_thread = None
        
        # Reset completion tracking
        self._download_completed = False
        self._completed_file_path = None
        
        # Clear the temp directory tracking so "Watch Now" can find the partial file
        if hasattr(self, '_download_temp_directory'):
            # Keep the temp directory for "Watch Now" functionality
            pass
        
        # Check for any remaining yt-dlp processes and kill them
        self._kill_remaining_yt_dlp_processes()
        
        # Re-enable the button
        self.button_watch_url.set_sensitive(True)
    
    def _kill_remaining_yt_dlp_processes(self):
        """Kill any remaining yt-dlp processes"""
        try:
            import subprocess
            
            # Try to find and kill yt-dlp processes on different platforms
            if os.name == 'nt':  # Windows
                try:
                    # Use taskkill to kill yt-dlp processes
                    result = subprocess.run(['taskkill', '/F', '/IM', 'yt-dlp.exe'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info("Successfully killed yt-dlp processes using taskkill")
                    else:
                        logger.warning(f"taskkill failed: {result.stderr}")
                except Exception as e:
                    logger.warning(f"Error using taskkill: {e}")
            else:  # Unix-like systems
                try:
                    # Use pkill to kill yt-dlp processes
                    result = subprocess.run(['pkill', '-f', 'yt-dlp'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info("Successfully killed yt-dlp processes using pkill")
                    else:
                        logger.warning(f"pkill failed: {result.stderr}")
                except Exception as e:
                    logger.warning(f"Error using pkill: {e}")
                    
                # Fallback: try to use killall
                try:
                    result = subprocess.run(['killall', 'yt-dlp'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info("Successfully killed yt-dlp processes using killall")
                except Exception as e:
                    logger.warning(f"Error using killall: {e}")
        except Exception as e:
            logger.warning(f"Error killing remaining yt-dlp processes: {e}")

    def _download_video_progressive(self, url: str):
        """Download video progressively with progress monitoring"""
        try:
            logger.info(f"=== STARTING DOWNLOAD FOR URL: {url} ===")
            from lada.gui.utils import download_video_progressive
            
            # Get estimated size first
            logger.info("Getting estimated video size...")
            estimated_size_mb = self._get_estimated_video_size(url)
            self._estimated_total_mb = estimated_size_mb
            
            logger.info(f"Estimated video size: {estimated_size_mb or 'None'} MB")
            
            # Get LADA's temp directory - simplified approach
            temp_directory = 'F:\\Pron'  # Use the configured temp directory directly for now
            logger.info(f"Using temp directory: {temp_directory}")
            
            # Store the temp directory for later file detection
            # Note: we'll update this to the actual temp directory when download starts
            self._download_temp_directory = None
            
            # Create a progress callback that includes estimated size
            def progress_callback(bytes_downloaded, formatted_progress=None):
                self._update_download_progress_with_size(bytes_downloaded, formatted_progress or "", estimated_size_mb)
            
            # Generate the expected file path immediately for "Watch Now" functionality
            # We know yt-dlp will create files in the temp directory
            import time
            stream_dir = pathlib.Path(temp_directory) / f"lada_stream_{int(time.time())}"
            expected_file_path = str(stream_dir / "stream.mp4")
            
            logger.info(f"Expected file path: {expected_file_path}")
            
            # Set the file path immediately for "Watch Now" functionality
            # Note: yt-dlp will create stream.mp4.part, so we need to account for that
            self._current_downloaded_file = expected_file_path + ".part"  # Account for yt-dlp adding .part
            self._download_temp_directory = str(stream_dir)
            logger.info(f"Set expected file path for Watch Now: {expected_file_path}")
            
            logger.info("About to call download_video_progressive...")
            
            # Create a list to hold the process reference and store as instance variable
            self._process_holder = [None]
            process_holder = self._process_holder
            logger.info(f"Created process holder: {process_holder}")
            
            # Start download (it will use the same path we just set)
            # Reset global cancellation flag for new download
            try:
                from lada.gui.utils import _download_cancelled
                _download_cancelled = False
                logger.info("Reset global download cancellation flag for new download")
            except ImportError:
                pass
            
            logger.info(f"Starting download with expected path: {expected_file_path}")
            downloaded_file, video_title = download_video_progressive(url, progress_callback, temp_directory, process_holder)
            
            # Store the process reference for cancellation
            self._current_process = process_holder[0]
            logger.info(f"Stored process reference: {self._current_process}, PID: {getattr(self._current_process, 'pid', 'No PID')}")
            logger.info(f"Process holder contents after download: {process_holder}")
            
            # Verify the process was stored
            if self._current_process is None:
                logger.error("WARNING: Process reference is None after download!")
            else:
                logger.info(f"Process reference confirmed: PID {self._current_process.pid}")
                
                # Log current process status
                logger.info(f"Current process status at start: {self._current_process.poll()}")
            
            logger.info(f"Download function returned: {downloaded_file}")
            logger.info(f"Type of returned value: {type(downloaded_file)}")
            
            # Handle cancellation case
            if downloaded_file == "":
                logger.info("Download was cancelled by user")
                # Reset the global cancellation flag
                try:
                    from lada.gui.utils import _download_cancelled
                    _download_cancelled = False
                except ImportError:
                    pass
                
                # Close the progress dialog and return without error - all on main thread
                def cleanup_ui():
                    try:
                        self._close_progress_dialog()
                        self.button_watch_url.set_sensitive(True)
                    except Exception as e:
                        logger.warning(f"Error during UI cleanup: {e}")
                
                GLib.idle_add(cleanup_ui)
                return
            
            if downloaded_file is None:
                logger.error("ERROR: download_video_progressive returned None!")
                raise Exception("Download function returned no file path")
            
            if not pathlib.Path(downloaded_file).exists():
                logger.error(f"ERROR: Downloaded file doesn't exist: {downloaded_file}")
                raise Exception(f"Downloaded file doesn't exist: {downloaded_file}")
            
            # Update with actual file path if different
            if downloaded_file and downloaded_file != expected_file_path:
                self._current_downloaded_file = downloaded_file
                downloaded_path = pathlib.Path(downloaded_file)
                self._download_temp_directory = str(downloaded_path.parent)
                logger.info(f"Updated file path after download: {downloaded_file}")
            
            # Store the downloaded file path for potential "Watch Now" functionality
            self._current_downloaded_file = downloaded_file
            
            # Set the temp directory to the parent directory of the downloaded file
            if downloaded_file:
                downloaded_path = pathlib.Path(downloaded_file)
                self._download_temp_directory = str(downloaded_path.parent)
                logger.info(f"Set temp directory to: {self._download_temp_directory}")
                
                # Rename video file to proper name if we detected a video title
                self._rename_downloaded_video(downloaded_file, video_title if 'video_title' in locals() else None)
            
            # Let the final progress callback handle the completion UI update
            # Don't close dialog here - let the callback do it when it processes the final update
            
            if downloaded_file:
                # Download successful - the final progress callback will handle UI updates
                logger.info(f"Progressive download completed: {downloaded_file}")
                
                # Reset global cancellation flag
                try:
                    from lada.gui.utils import _download_cancelled
                    _download_cancelled = False
                except ImportError:
                    pass
                
                # Store completion flag for the progress callback to handle
                self._download_completed = True
                
                # Create Gio.File object and emit signal (this will be done in the callback)
                self._completed_file_path = downloaded_file
            else:
                # Download failed or was cancelled
                logger.info("Download was cancelled or failed")
                
                # Reset global cancellation flag
                try:
                    from lada.gui.utils import _download_cancelled
                    _download_cancelled = False
                except ImportError:
                    pass
                
                # Don't show error dialog for cancelled downloads - clean UI on main thread
                def cleanup_ui():
                    try:
                        self._close_progress_dialog()
                        self.button_watch_url.set_sensitive(True)
                    except Exception as e:
                        logger.warning(f"Error during UI cleanup: {e}")
                
                GLib.idle_add(cleanup_ui)
                
        except Exception as e:
            logger.error(f"Failed to download video from URL {url}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Reset completion flag on error
            self._download_completed = False
            
            # Reset global cancellation flag
            try:
                from lada.gui.utils import _download_cancelled
                _download_cancelled = False
            except ImportError:
                pass
            
            # Check if this was a cancellation
            error_msg = str(e).lower()
            
            # Check for cancellation indicators in error message or global flag
            is_cancelled = (
                error_msg.strip() == "" or  # Empty error
                "download failed: unknown download error" in error_msg or  # Generic error from process termination (exact match)
                "progressive download failed" in error_msg and "download failed" in error_msg and "unknown download error" in error_msg or  # Full error message pattern
                "failed to download video" in error_msg and "unknown download error" in error_msg or  # Alternative pattern
                "terminated" in error_msg or  # Process termination
                "killed" in error_msg or  # Process killed
                "interrupted" in error_msg or  # Process interrupted
                error_msg.startswith("progressive download failed")  # Starts with our function name
            )
            
            # Also check the global flag as backup
            try:
                from lada.gui.utils import _download_cancelled
                if _download_cancelled:
                    is_cancelled = True
                    logger.info("Global cancellation flag detected")
            except ImportError:
                pass
            
            if is_cancelled:
                logger.info("Download was cancelled - not showing error dialog")
                
                # Clean UI operations on main thread
                def cleanup_ui():
                    try:
                        self._close_progress_dialog()
                        self.button_watch_url.set_sensitive(True)
                    except Exception as ui_e:
                        logger.warning(f"Error during UI cleanup: {ui_e}")
                
                GLib.idle_add(cleanup_ui)
                return
            
            # Capture the error message properly
            error_message = str(e)
            
            # Close progress dialog and show error on main thread
            def handle_error():
                try:
                    self._close_progress_dialog()
                    self._show_download_error_dialog(error_message)
                    self.button_watch_url.set_sensitive(True)
                except Exception as ui_e:
                    logger.warning(f"Error during error UI handling: {ui_e}")
            
            GLib.idle_add(handle_error)
    
    def _get_estimated_video_size(self, url: str) -> float | None:
        """Get estimated video size before download"""
        try:
            import subprocess
            import json
            
            info_cmd = [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--no-playlist",
                "--ignore-errors",
                url
            ]
            
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            
            if info_result.returncode == 0:
                video_info = json.loads(info_result.stdout)
                estimated_size = video_info.get('filesize') or video_info.get('filesize_approx')
                if estimated_size:
                    return estimated_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            pass  # Silent error handling
        
        return None
    
    def _update_download_progress_with_size(self, bytes_downloaded: int, formatted_progress: str | None = None, estimated_size_mb: float | None = None):
        """Update download progress with estimated size information and yt-dlp format"""
        # Don't log anything - just update UI
        
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            # Convert bytes to MB for display
            mb_downloaded = bytes_downloaded / (1024 * 1024)
            
            # Get estimated size from stored value if not provided
            if estimated_size_mb is None:
                estimated_size_mb = getattr(self, '_estimated_total_mb', None)
            
            # Determine if video is ready to watch (20MB threshold)
            watch_ready_threshold_mb = 20
            is_watch_ready = mb_downloaded >= watch_ready_threshold_mb
            
            def update_progress():
                try:
                    # Check if download is completed (no more progress expected)
                    is_completed = hasattr(self, '_download_completed') and self._download_completed
                    
                    # Update status label with yt-dlp format if available
                    if self._progress_label:
                        if formatted_progress and formatted_progress.startswith('[download]'):
                            # Use yt-dlp's original format
                            status_text = formatted_progress
                        elif is_completed:
                            status_text = f"Downloaded: {mb_downloaded:.1f} MB - Download complete!"
                        elif is_watch_ready:
                            status_text = f"Downloaded: {mb_downloaded:.1f} MB - Ready to watch!"
                        else:
                            status_text = f"Downloaded: {mb_downloaded:.1f} MB - Preparing for playback..."
                        self._progress_label.set_text(status_text)
                    
                    # Update size info
                    if self._size_label:
                        if estimated_size_mb and estimated_size_mb > 0:
                            percentage = int((mb_downloaded / estimated_size_mb) * 100)
                            size_text = f"{mb_downloaded:.1f} MB / {estimated_size_mb:.1f} MB ({percentage}%)"
                        else:
                            size_text = f"{mb_downloaded:.1f} MB downloaded"
                        self._size_label.set_text(size_text)
                    
                    # Enable/disable Watch Now button
                    if self._watch_now_button:
                        self._watch_now_button.set_sensitive(is_watch_ready)
                    
                    # Handle autostart playback if enabled and ready
                    if is_watch_ready and hasattr(self, '_autostart_checkbox') and self._autostart_checkbox:
                        if self._autostart_checkbox.get_active():
                            logger.info("Autostart enabled and file ready - starting playback automatically")
                            # Use GLib.idle_add to ensure this happens on main thread
                            GLib.idle_add(lambda: self._on_watch_now_clicked(self._watch_now_button))
                    
                    # If download is completed, close dialog and open video
                    if is_completed and hasattr(self, '_completed_file_path') and self._completed_file_path:
                        # Close progress dialog
                        self._close_progress_dialog()
                        
                        # Create Gio.File object and emit signal
                        gio_file = Gio.File.new_for_path(self._completed_file_path)
                        self.emit("files-selected", [gio_file])
                        
                        # Clean up completion flags
                        self._download_completed = False
                        self._completed_file_path = None
                        
                except Exception as e:
                    # Silent error handling - no logging
                    pass
                    
            # Use GLib.idle_add to ensure UI updates happen on main thread
            GLib.idle_add(update_progress)
        else:
            # Silently ignore callbacks when dialog is not available
            pass

    def _update_download_progress(self, bytes_downloaded: int, formatted_progress: str | None = None):
        """Update download progress (called from background thread)"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            # Convert bytes to MB for display
            mb_downloaded = bytes_downloaded / (1024 * 1024)
            
            # Store current downloaded file path for "Watch Now" functionality
            if hasattr(self, '_current_downloaded_file') and self._current_downloaded_file:
                # File path is already set from download function
                pass
            
            # Determine if video is ready to watch (20MB threshold)
            watch_ready_threshold_mb = 20
            is_watch_ready = mb_downloaded >= watch_ready_threshold_mb
            
            def update_progress():
                try:
                    # Update status label with yt-dlp format if available
                    if self._progress_label:
                        if formatted_progress and formatted_progress.startswith('[download]'):
                            # Use yt-dlp's original format
                            status_text = formatted_progress
                        elif is_watch_ready:
                            status_text = f"Downloaded: {mb_downloaded:.1f} MB - Ready to watch!"
                        else:
                            status_text = f"Downloaded: {mb_downloaded:.1f} MB - Preparing for playback..."
                        self._progress_label.set_text(status_text)
                    
                    # Update size info
                    if self._size_label:
                        if hasattr(self, '_estimated_total_mb') and self._estimated_total_mb:
                            percentage = int((mb_downloaded / self._estimated_total_mb) * 100)
                            size_text = f"{mb_downloaded:.1f} MB / {self._estimated_total_mb:.1f} MB ({percentage}%)"
                        else:
                            size_text = f"{mb_downloaded:.1f} MB downloaded"
                        self._size_label.set_text(size_text)
                    
                    # Enable/disable Watch Now button
                    if self._watch_now_button:
                        self._watch_now_button.set_sensitive(is_watch_ready)
                    
                    # Handle autostart playback if enabled and ready
                    if is_watch_ready and hasattr(self, '_autostart_checkbox') and self._autostart_checkbox:
                        if self._autostart_checkbox.get_active():
                            logger.info("Autostart enabled and file ready - starting playback automatically")
                            # Use GLib.idle_add to ensure this happens on main thread
                            GLib.idle_add(lambda: self._on_watch_now_clicked(self._watch_now_button))
                        
                except Exception as e:
                    pass  # Silent error handling
                    
            GLib.idle_add(update_progress)

    def _show_download_error_dialog(self, error_msg: str | None = None):
        """Show error dialog for download failure"""
        error_text = error_msg or "Failed to download video from the provided URL."
        dialog = Adw.AlertDialog.new("Download Failed", error_text)
        dialog.add_response("ok", "OK")
        dialog.set_default_response("ok")
        dialog.present()

    def _rename_downloaded_video(self, file_path: str, video_title: str | None):
        """Rename downloaded video to proper name based on video title"""
        try:
            if not video_title:
                logger.info("No video title available, skipping rename")
                return
                
            original_path = pathlib.Path(file_path)
            if not original_path.exists():
                logger.info(f"Original file doesn't exist: {original_path}")
                return
                
            # Clean video title for filename (remove invalid characters)
            safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            # Create new filename with same extension
            new_filename = f"{safe_title}{original_path.suffix}"
            new_path = original_path.parent / new_filename
            
            # Only rename if the new filename is different
            if new_path != original_path and not new_path.exists():
                original_path.rename(new_path)
                logger.info(f"Renamed video from {original_path.name} to {new_filename}")
                
                # Update the file path references
                self._current_downloaded_file = str(new_path)
                if hasattr(self, '_download_temp_directory'):
                    self._download_temp_directory = str(new_path.parent)
            else:
                logger.info("Skipping rename - file already exists or same name")
                
        except Exception as e:
            logger.warning(f"Error renaming video file: {e}")
    
    @GObject.Signal(name="files-selected", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass
