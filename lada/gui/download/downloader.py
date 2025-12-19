# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import threading
import urllib.parse
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlparse

import requests
from gi.repository import GLib, Gio

from lada.gui.config.config import Config
from .url_handler import URLHandlerFactory
from .progress_dialog import ProgressDialog
from .auth_dialog import AuthDialog
from .credentials import AuthCredentials

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation"""
    success: bool
    local_path: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0


class URLDownloader:
    """Main URL downloader class with comprehensive protocol support"""

    def __init__(self, config: Config, parent_widget):
        self.config = config
        self.parent = parent_widget
        self.active_downloads: Dict[str, threading.Thread] = {}

    def download_url(self, url: str, callback: Callable[[DownloadResult], None], show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """
        Download a URL and call callback with result

        Args:
            url: URL to download
            callback: Function to call with DownloadResult
            show_progress: Whether to show progress dialog
        """
        def do_download():
            try:
                # Parse URL and create appropriate handler
                parsed = urlparse(url)
                scheme = parsed.scheme.lower()

                if scheme in ('http', 'https'):
                    self._download_web_url(url, callback, show_progress, force_save_dialog)
                elif scheme in ('ftp', 'ftps'):
                    self._download_ftp_url(url, callback, show_progress, force_save_dialog)
                elif scheme == '' and url.startswith('\\\\'):
                    self._download_smb_path(url, callback, show_progress, force_save_dialog)
                elif scheme in ('file', 'smb'):
                    self._download_file_url(url, callback, show_progress, force_save_dialog)
                else:
                    result = DownloadResult(
                        success=False,
                        error_message=f"Unsupported URL scheme: {scheme}"
                    )
                    GLib.idle_add(lambda: callback(result))

            except Exception as e:
                logger.error(f"Download setup failed for {url}: {e}")
                result = DownloadResult(
                    success=False,
                    error_message=f"Failed to setup download: {str(e)}"
                )
                GLib.idle_add(lambda: callback(result))

        # Start download in background thread
        thread = threading.Thread(target=do_download, daemon=True)
        thread.start()

    def download_urls(self, urls: list[str], callback: Callable[[list[Gio.File]], None]) -> None:
        """
        Download URLs and call callback with all downloaded files

        Args:
            urls: List of URLs to download
            callback: Function to call with list of downloaded files
        """
        if not urls:
            GLib.idle_add(lambda: callback([]))
            return

        # Handle single vs multiple URLs differently
        if len(urls) == 1:
            # Single URL - always show save dialog for individual file location
            self.download_url(urls[0], lambda result: self._on_download_complete(result, callback), show_progress=True, force_save_dialog=True)
        else:
            # Multiple URLs - show download folder selection first, then restore folder
            self._show_batch_download_directory_dialog(urls, callback)

    def _show_batch_download_directory_dialog(self, urls: list[str], callback: Callable[[list[Gio.File]], None]) -> None:
        """Show directory selection dialog for download location"""
        from gi.repository import Gtk

        def on_dialog_result(dialog, result):
            try:
                selected = dialog.select_folder_finish(result)
                if selected is not None:
                    download_path = selected.get_path()
                    # Set download directory and proceed to restore directory selection
                    self.config.temp_directory = download_path
                    self._show_batch_export_directory_dialog(urls, callback)
                else:
                    # User cancelled - reset button sensitivity
                    self.parent.button_add_files.set_sensitive(True)
            except GLib.Error:
                # Error - reset button sensitivity
                self.parent.button_add_files.set_sensitive(True)

        file_dialog = Gtk.FileDialog()
        file_dialog.set_title("Select download folder")
        file_dialog.select_folder(callback=on_dialog_result)

    def _show_batch_export_directory_dialog(self, urls: list[str], callback: Callable[[list[Gio.File]], None]) -> None:
        """Show directory selection dialog for restoration location"""
        from gi.repository import Gtk

        def on_dialog_result(dialog, result):
            try:
                selected = dialog.select_folder_finish(result)
                if selected is not None:
                    export_path = selected.get_path()
                    # Set restore directory and proceed with downloads
                    self.config.export_directory = export_path
                    self._start_batch_downloads(urls, callback)
                else:
                    # User cancelled - reset button sensitivity
                    self.parent.button_add_files.set_sensitive(True)
            except GLib.Error:
                # Error - reset button sensitivity
                self.parent.button_add_files.set_sensitive(True)

        file_dialog = Gtk.FileDialog()
        file_dialog.set_title("Select restoration folder")
        file_dialog.select_folder(callback=on_dialog_result)

    def _start_batch_downloads(self, urls: list[str], callback: Callable[[list[Gio.File]], None]) -> None:
        """Start downloading multiple URLs in parallel with restoration"""
        downloaded_files = []
        completed_count = 0
        total_count = len(urls)

        def on_single_complete(result: DownloadResult):
            nonlocal completed_count
            if result.success and result.local_path:
                file_gio = Gio.File.new_for_path(result.local_path)
                downloaded_files.append(file_gio)
                # Add file to export queue immediately when download completes
                GLib.idle_add(lambda: self._add_downloaded_file_to_queue(file_gio))
            completed_count += 1

            # If all downloads completed, call the main callback
            if completed_count >= total_count:
                GLib.idle_add(lambda: callback(downloaded_files))

        # Start all downloads in parallel without progress dialogs
        for url in urls:
            self.download_url(url, on_single_complete, show_progress=False)

    def _add_downloaded_file_to_queue(self, file_gio: Gio.File) -> None:
        """Add a downloaded file to the export queue"""
        # This will trigger the export view to add the file and potentially start processing
        self.parent.emit("files-added", [file_gio])

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is supported by the downloader"""
        if not url or not url.strip():
            return False

        try:
            parsed = urlparse(url)
            # Check for supported schemes
            supported_schemes = {'http', 'https', 'ftp', 'ftps', 'file', 'smb'}
            if parsed.scheme.lower() in supported_schemes:
                return True

            # Check if it's a local file path (no scheme)
            if not parsed.scheme and (os.path.exists(url) or url.startswith('\\\\')):
                return True

            return False
        except Exception:
            return False

    def _show_invalid_urls_error(self, invalid_urls: list[str]) -> None:
        """Show error dialog for invalid URLs"""
        from gi.repository import Adw

        urls_text = "\n".join(f"• {url}" for url in invalid_urls[:5])  # Show max 5
        if len(invalid_urls) > 5:
            urls_text += f"\n• ... and {len(invalid_urls) - 5} more"

        error_dialog = Adw.AlertDialog(
            heading="Invalid URLs",
            body=f"The following URLs are not valid or supported:\n\n{urls_text}\n\nSupported formats:\n• http:// and https:// URLs\n• ftp:// and ftps:// URLs\n• file:// paths\n• Local file paths\n• SMB paths (\\\\server\\share)",
        )
        error_dialog.add_response("ok", "OK")
        error_dialog.choose(self.parent, None, lambda d, t: None)

    def _reset_to_initial_state(self) -> None:
        """Reset the application to initial state (file selection view)"""
        # Clear the export queue
        self.parent.model.remove_all()

        # Find the main window and switch to file selection view
        # Use GTK's root finding to get the toplevel window
        toplevel = self.parent.get_root()
        if hasattr(toplevel, 'stack') and hasattr(toplevel.stack, 'set_visible_child_name'):
            try:
                toplevel.stack.set_visible_child_name("file-selection")
            except Exception as e:
                logger.warning(f"Failed to reset to initial state: {e}")

        # Reset button sensitivity
        self.parent.button_add_files.set_sensitive(True)

    def show_url_input_dialog(self, callback: Callable[[list[Gio.File]], None]) -> None:
        """
        Show URL input dialog and download the URLs

        Args:
            callback: Function to call with downloaded files
        """
        from gi.repository import Gtk, Adw

        dialog = Adw.AlertDialog(
            heading="Add Video URLs",
            body="Enter one or more video URLs (one per line) to add for restoration:",
        )

        text_view = Gtk.TextView()
        text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        text_view.set_hexpand(True)
        text_view.set_vexpand(True)
        text_view.set_size_request(400, 100)

        buffer = text_view.get_buffer()
        buffer.set_text("")  # Start with empty text area

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_child(text_view)
        scrolled.set_size_request(400, 100)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.append(scrolled)
        dialog.set_extra_child(box)

        def on_response(dialog, task):
            try:
                response = dialog.choose_finish(task)
                if response == "add":
                    start_iter, end_iter = buffer.get_bounds()
                    text = buffer.get_text(start_iter, end_iter, False).strip()
                    urls = [url.strip() for url in text.split('\n') if url.strip()]
                    if urls:
                        # Validate URLs before processing
                        invalid_urls = []
                        for url in urls:
                            if not self._is_valid_url(url):
                                invalid_urls.append(url)

                        if invalid_urls:
                            # Show error dialog for invalid URLs
                            self._show_invalid_urls_error(invalid_urls)
                            return  # Stay in dialog

                        # All URLs valid, proceed
                        self.download_urls(urls, callback)
                    else:
                        # No URLs entered - reset to initial state
                        self._reset_to_initial_state()
                else:
                    # Cancelled - reset to initial state
                    self._reset_to_initial_state()
            except GLib.Error:
                # Error - reset to initial state
                self._reset_to_initial_state()

        def on_closed(dialog):
            # Reset button sensitivity when dialog is closed
            self.parent.button_add_files.set_sensitive(True)

        dialog.add_response("cancel", "Cancel")
        dialog.add_response("add", "Restore")
        dialog.set_response_appearance("add", Adw.ResponseAppearance.SUGGESTED)
        dialog.connect("closed", on_closed)
        dialog.choose(self.parent, None, on_response)

    def _on_download_complete(self, result: DownloadResult, callback: Callable[[list[Gio.File]], None]) -> None:
        """Handle download completion"""
        if result.success and result.local_path:
            # Add downloaded file to export queue
            file_gio = Gio.File.new_for_path(result.local_path)
            callback([file_gio])
        else:
            # Show error dialog
            self._show_download_error(result.error_message or "Unknown error")

    def _download_web_url(self, url: str, callback: Callable[[DownloadResult], None], show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """Download HTTP/HTTPS URL with validation"""
        try:
            # HEAD request to validate
            head_response = requests.head(url, timeout=10, allow_redirects=True)
            head_response.raise_for_status()

            content_type = head_response.headers.get('Content-Type', '')
            if not content_type.startswith('video/'):
                result = DownloadResult(
                    success=False,
                    error_message=f"The URL does not point to a video file. Content-Type: {content_type}"
                )
                GLib.idle_add(lambda: callback(result))
                return

            # Get filename and proceed
            filename = self._extract_filename(url, head_response)
            content_length = head_response.headers.get('Content-Length')

            self._proceed_with_download(url, filename, content_length, None, callback, show_progress, force_save_dialog)

        except requests.exceptions.RequestException as e:
            result = DownloadResult(
                success=False,
                error_message="Failed to access the URL. Please check the URL and your internet connection."
            )
            GLib.idle_add(lambda: callback(result))
        except Exception as e:
            result = DownloadResult(
                success=False,
                error_message="Failed to analyze the URL. Please try again."
            )
            GLib.idle_add(lambda: callback(result))

    def _download_ftp_url(self, url: str, callback: Callable[[DownloadResult], None], show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """Download FTP URL with authentication"""
        # Show auth dialog
        auth_dialog = AuthDialog(self.parent)
        auth_dialog.connect("auth-provided", lambda d, creds: self._download_ftp_with_auth(url, creds, callback, show_progress, force_save_dialog))
        auth_dialog.connect("cancelled", lambda d: callback(DownloadResult(success=False, error_message="FTP authentication cancelled")))
        auth_dialog.show()

    def _download_ftp_with_auth(self, url: str, creds: AuthCredentials, callback: Callable[[DownloadResult], None], show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """Download FTP URL with provided credentials"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)

            # Try HEAD request with auth
            try:
                auth = (creds.username, creds.password) if creds.username and creds.password else None
                head_response = requests.head(url, auth=auth, timeout=10)
                head_response.raise_for_status()
                content_length = head_response.headers.get('Content-Length')
            except:
                content_length = None

            # Get filename
            filename = os.path.basename(parsed.path)
            if not filename or '.' not in filename:
                filename = 'downloaded_video.mp4'

            # Proceed with download
            self._proceed_with_download(url, filename, content_length, creds, callback, show_progress, force_save_dialog)

        except requests.exceptions.RequestException as e:
            result = DownloadResult(
                success=False,
                error_message="Failed to access FTP server. Please check credentials and URL."
            )
            GLib.idle_add(lambda: callback(result))
        except Exception as e:
            result = DownloadResult(
                success=False,
                error_message="Failed to connect to FTP server."
            )
            GLib.idle_add(lambda: callback(result))

    def _download_smb_path(self, path: str, callback: Callable[[DownloadResult], None], show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """Handle Windows SMB path"""
        try:
            import urllib.parse
            file_url = urllib.parse.urljoin('file:', urllib.parse.quote(path.replace('\\', '/')))
            self._download_file_url(file_url, callback, show_progress, force_save_dialog)
        except Exception as e:
            result = DownloadResult(
                success=False,
                error_message=f"Failed to process SMB path: {str(e)}"
            )
            GLib.idle_add(lambda: callback(result))

    def _download_file_url(self, url: str, callback: Callable[[DownloadResult], None], show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """Handle local file URLs"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme == 'file':
                file_path = urllib.parse.unquote(parsed.path)
                if os.name == 'nt' and file_path.startswith('/'):
                    file_path = file_path[1:]  # Remove leading slash on Windows

                if os.path.exists(file_path) and os.path.isfile(file_path):
                    result = DownloadResult(
                        success=True,
                        local_path=file_path,
                        file_size=os.path.getsize(file_path)
                    )
                    GLib.idle_add(lambda: callback(result))
                else:
                    result = DownloadResult(
                        success=False,
                        error_message="File not found or not accessible."
                    )
                    GLib.idle_add(lambda: callback(result))
            else:
                result = DownloadResult(
                    success=False,
                    error_message="Unsupported file URL scheme"
                )
                GLib.idle_add(lambda: callback(result))
        except Exception as e:
            result = DownloadResult(
                success=False,
                error_message=f"Failed to access the file: {str(e)}"
            )
            GLib.idle_add(lambda: callback(result))

    def _extract_filename(self, url: str, response) -> str:
        """Extract filename from response headers or URL"""
        filename = None
        if 'Content-Disposition' in response.headers:
            import re
            match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', response.headers['Content-Disposition'])
            if match:
                filename = match.group(1).strip('\'"')

        if not filename:
            filename = os.path.basename(url.split('?')[0])

        content_type = response.headers.get('Content-Type', '')
        if not filename or '.' not in filename:
            import mimetypes
            ext = mimetypes.guess_extension(content_type) or '.mp4'
            if not filename:
                filename = f'downloaded_video{ext}'
            else:
                filename = f"{filename}{ext}"

        return filename

    def _proceed_with_download(self, url: str, filename: str, content_length: Optional[str],
                               auth: Optional[AuthCredentials], callback: Callable[[DownloadResult], None],
                               show_progress: bool = True, force_save_dialog: bool = False) -> None:
        """Determine download and restore paths, then start download"""
        if self.config.export_directory and not force_save_dialog:
            # Use export directory for restored file, download to temp
            temp_dir = self.config.temp_directory
            download_path = os.path.join(temp_dir, filename)
            restored_path = os.path.join(self.config.export_directory, self._get_restored_filename(filename))
            self._download_to_path(url, download_path, restored_path, content_length, auth, callback, show_progress)
        else:
            # Show save dialog for the restored file location
            self._show_save_dialog_for_restored_file(url, filename, content_length, auth, callback, show_progress)

    def _get_restored_filename(self, original_filename: str) -> str:
        """Generate restored filename based on config pattern"""
        base_name = os.path.splitext(original_filename)[0]
        return self.config.file_name_pattern.replace("{orig_file_name}", base_name)

    def _download_to_path(self, url: str, download_path: str, restore_path: str,
                          content_length: Optional[str], auth: Optional[AuthCredentials],
                          callback: Callable[[DownloadResult], None], show_progress: bool = True) -> None:
        """Download URL to specified path with optional progress dialog"""
        progress_dialog = None
        if show_progress:
            progress_dialog = ProgressDialog(self.parent, url, content_length)
            progress_dialog.connect("cancelled", lambda: self._cancel_download())

        def on_download_complete():
            if progress_dialog:
                progress_dialog.close()
            result = DownloadResult(
                success=True,
                local_path=download_path,
                file_size=os.path.getsize(download_path) if os.path.exists(download_path) else 0
            )
            callback(result)

        def on_download_error(error_msg: str):
            if progress_dialog:
                progress_dialog.close()
            result = DownloadResult(success=False, error_message=error_msg)
            callback(result)

        # Start the actual download
        self._perform_download(url, download_path, content_length, auth,
                              progress_dialog, on_download_complete, on_download_error)

    def _perform_download(self, url: str, output_path: str, content_length: Optional[str],
                          auth: Optional[AuthCredentials], progress_dialog: Optional[ProgressDialog],
                          on_complete: Callable, on_error: Callable[[str], None]) -> None:
        """Perform the actual download with optional progress tracking"""
        def download_worker():
            try:
                # Prepare request
                request_kwargs = {'stream': True, 'timeout': 30}
                if auth:
                    request_kwargs['auth'] = (auth.username, auth.password)

                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                total_size = int(content_length) if content_length else None
                downloaded = 0

                if progress_dialog:
                    GLib.idle_add(lambda: progress_dialog.set_status("Downloading..."))

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if progress_dialog and hasattr(progress_dialog, 'cancelled') and progress_dialog.cancelled:
                            break
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_dialog and hasattr(progress_dialog, 'update_progress') and total_size:
                                progress = downloaded / total_size
                                GLib.idle_add(lambda: progress_dialog.update_progress(progress, downloaded, total_size))

                if progress_dialog and progress_dialog.cancelled:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return

                # Verify download
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    os.remove(output_path)
                    raise Exception("Downloaded file is empty")

                GLib.idle_add(on_complete)

            except requests.exceptions.RequestException as e:
                GLib.idle_add(lambda: on_error("Failed to download the video from the provided URL. Please check the URL and your internet connection."))
            except Exception as e:
                GLib.idle_add(lambda: on_error("Failed to download the video from the provided URL. Please check the URL and try again."))

        thread = threading.Thread(target=download_worker, daemon=True)
        thread.start()

    def _show_save_dialog_for_restored_file(self, url: str, filename: str, content_length: Optional[str],
                                           auth: Optional[AuthCredentials], callback: Callable[[DownloadResult], None],
                                           show_progress: bool = True) -> None:
        """Show save dialog for the restored file location"""
        from gi.repository import Gtk

        def on_dialog_result(dialog, result):
            try:
                selected = dialog.save_finish(result)
                if selected is not None:
                    restore_path = selected.get_path()

                    # Download to temp location
                    temp_dir = self.config.temp_directory
                    download_path = os.path.join(temp_dir, filename)

                    GLib.idle_add(lambda: self._download_to_path(url, download_path, restore_path, content_length, auth, callback, show_progress))
            except GLib.Error as error:
                if error.code == 2:  # Dismissed by user
                    pass
                else:
                    logger.error(f"Save dialog error: {error}")

        file_dialog = Gtk.FileDialog()
        video_file_filter = Gtk.FileFilter()
        video_file_filter.add_mime_type("video/*")
        file_dialog.set_default_filter(video_file_filter)
        file_dialog.set_title("Save restored video file")
        restored_filename = self._get_restored_filename(filename)
        file_dialog.set_initial_name(restored_filename)
        file_dialog.save(callback=on_dialog_result)

    def _cancel_download(self) -> None:
        """Cancel the current download"""
        logger.info("Download cancelled by user")

    def _show_download_error(self, message: str) -> None:
        """Show download error dialog"""
        from gi.repository import Adw

        error_dialog = Adw.AlertDialog(
            heading="Download Failed",
            body=message,
        )
        error_dialog.add_response("ok", "OK")
        error_dialog.choose(self.parent, None, lambda d, t: None)