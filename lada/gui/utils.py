# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from gettext import gettext as _
import shutil

import torch
from gi.repository import Gio
from gi.repository import Gtk, GLib, Gdk

from lada import LOG_LEVEL
from lada.gui.config.config import Config
from lada.utils import video_utils
from lada.utils.video_utils import EncodingPreset

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def is_device_available(device: str) -> bool:
    device = device.lower()
    if device == 'cpu':
        return True
    elif device.startswith("cuda:"):
        gpu_id = device_to_gpu_id(device)
        return gpu_id is not None and gpu_id < torch.cuda.device_count()
    return False


def device_to_gpu_id(device) -> int | None:
    if device.startswith("cuda:"):
        return int(device.split(":")[-1])
    return None


def get_available_gpus():
    gpus = []
    gpu_count = torch.cuda.device_count()
    for id in range(gpu_count):
        gpu_name = torch.cuda.get_device_properties(id).name
        # We're using these GPU names in a ComboBox but libadwaita sets up the label with max-width-chars: 20 and there does not
        # seem to be a way to overwrite this. So let's try to make sure GPU names are below 20 characters to be readable
        if gpu_name.startswith("NVIDIA GeForce RTX"):
            gpu_name = gpu_name.replace("NVIDIA GeForce RTX", "RTX")
        gpus.append((id, gpu_name))
    return gpus

def skip_if_uninitialized(f):
    def noop(*args):
        return
    def wrapper(*args):
        return f(*args) if args[0].init_done else noop
    return wrapper

def set_validation_css_classes(widget: Gtk.Widget, is_valid: bool):
    focused = "focused" in widget.get_css_classes()
    all_classes = {"success", "warning", "error"}
    def add_if_not_present(class_name):
        if class_name not in widget.get_css_classes():
            for other_class_names in all_classes.difference({class_name}):
                widget.remove_css_class(other_class_names)
            if class_name:
                widget.add_css_class(class_name)
    if is_valid:
        if focused:
            add_if_not_present("success")
        else:
            add_if_not_present(None)
    else:
        if focused:
            add_if_not_present("warning")
        else:
            add_if_not_present("error")

def validate_file_name_pattern(file_name_pattern: str) -> bool:
    if not "{orig_file_name}" in file_name_pattern:
        return False
    if os.sep in file_name_pattern:
        return False
    file_extension = os.path.splitext(file_name_pattern)[1].lower()
    if file_extension not in [".mp4", ".mkv", ".mov", ".m4v"]:
        return False
    return True

def validate_preset_description(description: str, config: Config, original_description: str | None) -> bool:
    presets = []
    presets.extend(config.custom_encoding_presets if original_description is None else [p for p in config.custom_encoding_presets if p.description != original_description])
    presets.extend(video_utils.get_encoding_presets())
    presets_descriptions = [preset.description.lower() for preset in presets]
    return description.lower() not in presets_descriptions

def filter_video_files(files: list[Gio.File]) -> list[Gio.File]:
    def is_video_file(file: Gio.File):
        file_info: Gio.FileInfo = file.query_info("standard::content-type", Gio.FileQueryInfoFlags.NONE)
        content_type = file_info.get_content_type() # on linux content_type is MIME type but on windows it's just a file extension
        if content_type is None: return False
        mime_type = Gio.content_type_get_mime_type(content_type)
        if mime_type is None: return False
        return mime_type.startswith("video/")
    filtered_files = [file for file in files if is_video_file(file)]
    return filtered_files

def show_open_files_dialog(callback, dismissed_callback):
    file_dialog = Gtk.FileDialog()
    video_file_filter = Gtk.FileFilter()
    video_file_filter.add_mime_type("video/*")
    file_dialog.set_default_filter(video_file_filter)
    file_dialog.set_title(_("Select one or multiple video files"))
    def on_open_multiple(_file_dialog, result):
        try:
            video_files = _file_dialog.open_multiple_finish(result)
            if len(video_files) > 0:
                callback(video_files)
        except GLib.Error as error:
            if error.code == 2: # "Dismissed by user"
                dismissed_callback()
                logger.debug("FileDialog cancelled: Dismissed by user")
            else:
                logger.error(f"Error opening file: {error.message}")
                raise error
    file_dialog.open_multiple(callback=on_open_multiple)

def create_video_files_drop_target(callback):
    drop_target = Gtk.DropTarget.new(Gio.File, actions=Gdk.DragAction.COPY)
    drop_target.set_gtypes((Gdk.FileList,))
    def on_file_drop(_drop_target, files: list[Gio.File], x, y):
        video_files = filter_video_files(files)
        if len(video_files) > 0:
            callback(video_files)
    drop_target.connect("drop", on_file_drop)
    return drop_target

def translate_ui_xml(path: str) -> str:
    with open(path, 'r', encoding="utf-8") as file:
        element = file.read()
    tree = ET.fromstring(element)
    for node in tree.iter():
        if 'translatable' in node.attrib and node.text:
            node.text = _(node.text)
            del node.attrib["translatable"]
    as_str = ET.tostring(tree, encoding='utf-8', method='xml')
    return as_str

def dump_encoder_options(encoder: str) -> str:
    result = subprocess.run(["ffmpeg", "-loglevel", "quiet", "-h", f"encoder={encoder}"], capture_output=True, text=True)
    text = result.stdout.strip().replace("Exiting with exit code 0", "").strip()
    return text

def get_next_custom_preset(config: Config) -> EncodingPreset:
    num = len(config.custom_encoding_presets) + 1
    description = _("Custom Preset {custom_preset_num}").format(custom_preset_num=num)
    return EncodingPreset(f"custom-preset-{num}", description, True, "libx264", "")

def get_selected_preset(config: Config) -> EncodingPreset:
    presets = []
    presets.extend(video_utils.get_encoding_presets())
    presets.extend(config.custom_encoding_presets)
    for preset in presets:
        if preset.name == config.encoding_preset_name:
            return preset
    raise ValueError("Selected preset not found")

def get_preset_by_name(config: Config, name: str) -> EncodingPreset:
    presets = []
    presets.extend(video_utils.get_encoding_presets())
    presets.extend(config.custom_encoding_presets)
    for preset in presets:
        if preset.name == name:
            return preset
    raise ValueError("Invalid preset name")

def is_unique_preset_description(description) -> bool:
    return not any([preset.description == description for preset in video_utils.get_encoding_presets()])

def download_video_with_ytdlp(url: str) -> list[str]:
    """Download video from URL using yt-dlp
    
    Args:
        url: Video URL to download
        
    Returns:
        List of downloaded file paths
        
    Raises:
        Exception: If download fails
    """
    import subprocess
    import json
    import atexit
    import shutil
    
    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix="lada_ytdlp_"))
    
    # Register cleanup function
    def cleanup_temp_dir():
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    atexit.register(cleanup_temp_dir)
    
    try:
        # First, get video info to check if it's downloadable
        info_cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            url
        ]
        
        logger.info(f"Getting video info for: {url}")
        info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(info_result.stdout)
        
        # Check if video has formats available
        if not video_info.get('formats'):
            raise Exception("No downloadable formats available for this video")
        
        # Try progressive streaming first (m3u8, etc.)
        progressive_formats = []
        for fmt in video_info.get('formats', []):
            if fmt.get('protocol') in ['http', 'https'] and fmt.get('ext') in ['mp4', 'webm']:
                progressive_formats.append(fmt)
        
        # If we have progressive formats, use the best one
        if progressive_formats:
            # Sort by quality (prefer mp4, then by height and bitrate)
            progressive_formats.sort(key=lambda x: (
                x.get('ext') == 'mp4',  # Prefer mp4
                x.get('height', 0) if x.get('height') else 0,  # Then by height
                x.get('tbr', 0) if x.get('tbr') else 0  # Then by bitrate
            ), reverse=True)
            best_format = progressive_formats[0]
            format_id = best_format['format_id']
            logger.info(f"Using progressive format: {format_id} ({best_format.get('ext')} {best_format.get('height', '?')}p)")
        else:
            # Fall back to best quality with ffmpeg merging
            format_id = "best[height<=1080]/best"
            logger.info(f"Using format: {format_id}")
        
        # Download the video
        output_template = str(temp_dir / "%(title)s.%(ext)s")
        download_cmd = [
            "yt-dlp",
            "-f", format_id,
            "-o", output_template,
            "--no-playlist",
            "--ignore-errors",
            url
        ]
        
        logger.info(f"Starting download: {url}")
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown download error"
            if "This video is not available" in error_msg:
                raise Exception("Video is not available or requires authentication")
            elif "Unable to download webpage" in error_msg:
                raise Exception("Unable to access video page")
            else:
                raise Exception(f"Download failed: {error_msg}")
        
        # Find downloaded files
        downloaded_files = []
        for file_path in temp_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.m4v']:
                downloaded_files.append(str(file_path))
        
        if not downloaded_files:
            raise Exception("No video file was downloaded")
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} file(s)")
        return downloaded_files
        
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp command failed: {e}")
        raise Exception(f"yt-dlp error: {e.stderr or str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse video info: {e}")
        raise Exception("Failed to get video information")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Clean up temp directory on failure
        cleanup_temp_dir()
        raise

def _is_valid_url(url: str) -> bool:
    """Check if URL has valid format"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def _is_likely_supported_site(url: str) -> bool:
    """Check if URL is from a site that's likely supported by yt-dlp"""
    # List of commonly supported sites
    supported_sites = [
        'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
        'facebook.com', 'instagram.com', 'tiktok.com', 'twitch.tv',
        'twitter.com', 'reddit.com', 'bilibili.com', 'niconico.jp',
        'soundcloud.com', 'bandcamp.com', 'mixcloud.com', 'archive.org'
    ]
    
    url_lower = url.lower()
    for site in supported_sites:
        if site in url_lower:
            return True
    
    return False

# Global flag to track download cancellation
_download_cancelled = False

# Track active processes for cancellation
_active_processes = {}

def download_video_progressive(url: str, progress_callback=None, temp_directory=None, process_holder=None) -> tuple[str, str | None]:
    """Download video progressively for immediate playback
    
    Args:
        url: Video URL to download
        progress_callback: Optional callback for progress updates
        temp_directory: Optional temp directory path (uses LADA config if not provided)
        process_holder: Optional reference to store the process object for cancellation
        
    Returns:
        Tuple of (path to downloadable file, video title) that can be played while downloading
        
    Raises:
        Exception: If download fails
    """
    import subprocess
    import json
    import os
    import threading
    import re
    from pathlib import Path
    import time as time_module
    
    # Get temp directory
    if temp_directory:
        temp_base_dir = temp_directory
    else:
        # Use LADA's default temp directory
        temp_base_dir = tempfile.gettempdir()
    
    # Create temporary file in LADA's temp directory
    temp_dir = Path(temp_base_dir) / f"lada_stream_{int(time_module.time())}"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "stream.mp4"
    
    # Declare global variables
    global _download_cancelled
    
    # Check for cancellation immediately
    if _download_cancelled:
        logger.info("Download cancelled before starting")
        return ("", None)
    
    try:
        # Check for cancellation again
        if _download_cancelled:
            logger.info("Download cancelled during setup")
            return ("", None)
            
        # First, validate the URL format
        if not _is_valid_url(url):
            raise Exception("Invalid URL format. Please enter a complete URL starting with http:// or https://")
        
        # Check if URL is from a potentially supported site
        if not _is_likely_supported_site(url):
            logger.warning(f"URL may be from unsupported site: {url}")
            # Continue anyway as yt-dlp might support it
        
        # Get video info first to check if it's downloadable
        info_cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            "--ignore-errors",
            url
        ]
        
        logger.info(f"Getting video info for: {url}")
        info_result = subprocess.run(info_cmd, capture_output=True, text=True)
        
        if info_result.returncode != 0:
            error_msg = info_result.stderr.strip()
            
            # Provide specific error messages for common issues
            if "unsupported URL" in error_msg.lower():
                raise Exception("This website is not supported by yt-dlp. Please try videos from supported sites like:\n• YouTube (youtube.com)\n• Vimeo (vimeo.com)\n• Dailymotion (dailymotion.com)\n• Facebook (facebook.com)\n• Twitch (twitch.tv)\n• And many more!")
            elif "video is unavailable" in error_msg.lower():
                raise Exception("This video is not available. It may have been removed, made private, or require authentication.")
            elif "no video formats found" in error_msg.lower():
                raise Exception("No downloadable video formats found for this URL. The video might be live, private, or region-restricted.")
            elif "unable to download webpage" in error_msg.lower():
                raise Exception("Unable to access the video page. Please check the URL and try again. The site might be blocking automated access.")
            elif "sign in" in error_msg.lower() or "login" in error_msg.lower():
                raise Exception("This video requires sign-in or authentication. Please use a public video or try a different URL.")
            elif "age" in error_msg.lower() and "restricted" in error_msg.lower():
                raise Exception("This video is age-restricted. Please try a different video or use a supported public video platform.")
            else:
                # Check for common patterns in error messages
                if "403" in error_msg or "forbidden" in error_msg.lower():
                    raise Exception("Access denied. This video might be geo-restricted or require special permissions.")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    raise Exception("Video not found. Please check the URL and try again.")
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    raise Exception("Rate limit exceeded. Please wait a few minutes and try again.")
                else:
                    raise Exception(f"Unable to extract video information from this URL. Please try a different video or use a supported platform like YouTube, Vimeo, or Dailymotion.")
        
        # Parse video info
        try:
            video_info = json.loads(info_result.stdout)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse video information: {str(e)}")
        
        # Get video title for logging
        video_title = video_info.get('title', 'Unknown Video')
        logger.info(f"Found video: {video_title}")
        
        # Get estimated file size if available
        estimated_size = video_info.get('filesize') or video_info.get('filesize_approx')
        estimated_size_mb = None
        if estimated_size:
            estimated_size_mb = estimated_size / (1024*1024)
            logger.info(f"Video estimated size: {estimated_size_mb:.1f} MB")
        
        # Find best streaming-friendly format (prioritize higher quality)
        # Prioritize formats with video and audio, up to 1080p max
        streaming_formats = []
        for fmt in video_info.get('formats', []):
            if (fmt.get('protocol') in ['http', 'https'] and 
                fmt.get('ext') in ['mp4', 'webm'] and
                fmt.get('vcodec') != 'none' and  # Has video
                fmt.get('acodec') != 'none' and  # Has audio
                fmt.get('height') and fmt.get('height') <= 1080):  # Max 1080p for stability
                streaming_formats.append(fmt)
        
        if not streaming_formats:
            # Fall back to any available format with audio up to 1080p
            for fmt in video_info.get('formats', []):
                if (fmt.get('protocol') in ['http', 'https'] and 
                    fmt.get('ext') in ['mp4', 'webm'] and
                    fmt.get('acodec') != 'none' and  # Has audio
                    (not fmt.get('height') or fmt.get('height') <= 1080)):
                    streaming_formats.append(fmt)
        
        if not streaming_formats:
            # Final fallback - any format
            for fmt in video_info.get('formats', []):
                if fmt.get('protocol') in ['http', 'https'] and fmt.get('ext') in ['mp4', 'webm']:
                    streaming_formats.append(fmt)
        
        if not streaming_formats:
            raise Exception("No progressive streaming format available. This video may require download and processing rather than streaming. Try a different video with streaming support.")
        
        # Sort by quality and compatibility (prefer mp4, higher resolution, then by bitrate)
        streaming_formats.sort(key=lambda x: (
            x.get('ext') == 'mp4',  # Prefer mp4 for better compatibility
            x.get('height', 0) if x.get('height') else 0,  # Prefer HIGHER resolution for quality
            x.get('tbr', 0) if x.get('tbr') else 0  # Then by bitrate
        ), reverse=True)
        
        best_format = streaming_formats[0]
        format_id = best_format['format_id']
        
        # Update extension based on format
        ext = best_format.get('ext', 'mp4')
        
        # Use regular file (yt-dlp will add .part automatically)
        temp_file = temp_dir / f"stream.{ext}"
        # yt-dlp automatically adds .part to the filename, so we don't need to add it
        
        logger.info(f"Starting progressive download: {format_id} ({ext} {best_format.get('height', '?')}p)")
        
        # Check for cancellation before starting process
        if _download_cancelled:
            logger.info("Download cancelled before starting process")
            return ("", None)
        
        # Start download process with settings optimized to minimize I/O contention
        cmd = [
            "yt-dlp",
            "-f", format_id,
            "-o", str(temp_file),
            "--no-playlist",
            "--ignore-errors",
            "--fragment-retries", "1",  # Quick retries
            "-N", "1",  # Single-threaded for I/O efficiency
            "--no-check-certificates",  # Skip certificate checks
            "--buffer-size", "16M",  # Large buffer for smooth I/O
            "--http-chunk-size", "20M",  # Large chunks for efficiency
            url
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        
        logger.info(f"Created yt-dlp process with PID: {process.pid}")
        
        # Store process reference for cancellation
        if process_holder is not None:
            process_holder[0] = process
            logger.info(f"Stored process reference in holder: {process_holder[0]}")
        else:
            logger.warning("process_holder is None, cannot store process reference")
        
        # Shared variable for tracking progress across thread
        last_reported_size = [0]  # Use list to make it mutable across thread
        
        # Report initial progress (download starting)
        if progress_callback:
            progress_callback(0)
        
        # Start progress monitoring immediately
        if progress_callback:
            def monitor_progress():
                try:
                    import re
                    
                    # Regular expression to match yt-dlp progress format
                    # Matches patterns like: "[download] 7.1% of 638.44MiB at 1.29MiB/s ETA 07:40"
                    progress_pattern = re.compile(r'\[download\]\s+([\d.]+)%\s+of\s+([\d.]+)([KMGT]?i?B)\s+at\s+([\d.]+)([KMGT]?i?B/s)\s+ETA\s+([\d:]+)')
                    
                    # Ensure stdout is available
                    if not process.stdout:
                        logger.error("Process stdout is None, cannot monitor progress")
                        return
                    
                    for line in iter(process.stdout.readline, ''):
                        # Check global cancellation flag
                        if _download_cancelled:
                            logger.info("Download cancelled via global flag, stopping progress monitoring")
                            break
                            
                        if process.poll() is not None:
                            logger.info("Process terminated, stopping progress monitoring")
                            break
                            
                        line = line.strip()
                        
                        # Look for yt-dlp progress lines
                        if '[download]' in line:
                            match = progress_pattern.search(line)
                            if match:
                                try:
                                    percentage = float(match.group(1))
                                    total_size_str = match.group(2) + match.group(3)
                                    speed_str = match.group(4) + match.group(5)
                                    eta_str = match.group(6)
                                    
                                    # Extract total size in bytes for our callback
                                    total_size_num = float(match.group(2))
                                    total_size_unit = match.group(3)
                                    
                                    # Convert to bytes with proper handling
                                    unit_multipliers = {
                                        'B': 1, 
                                        'KB': 1024, 
                                        'MB': 1024**2, 
                                        'GB': 1024**3,
                                        'KiB': 1024, 
                                        'MiB': 1024**2, 
                                        'GiB': 1024**3
                                    }
                                    
                                    # Handle both decimal and binary units
                                    unit = total_size_unit
                                    if unit in unit_multipliers:
                                        total_size_bytes = int(total_size_num * unit_multipliers[unit])
                                    else:
                                        # Fallback: assume MiB if unit not recognized
                                        total_size_bytes = int(total_size_num * 1024 * 1024)
                                    
                                    downloaded_bytes = int(total_size_bytes * percentage / 100)
                                    
                                    # Ensure we have a valid bytes value
                                    if downloaded_bytes <= 0:
                                        # Fallback: use file size if available
                                        part_file = temp_dir / f"{temp_file.name}.part"
                                        if part_file.exists():
                                            downloaded_bytes = part_file.stat().st_size
                                            logger.debug(f"Using file size fallback: {downloaded_bytes} bytes")
                                        else:
                                            # Estimate based on percentage of known size
                                            if estimated_size_mb:
                                                downloaded_bytes = int(estimated_size_mb * 1024 * 1024 * percentage / 100)
                                            else:
                                                downloaded_bytes = int(percentage * 1024 * 1024)  # Assume 1GB file
                                    
                                    # Create formatted progress string
                                    formatted_progress = f"[download] {percentage:5.1f}% of {total_size_str} at {speed_str} ETA {eta_str}"
                                    
                                    # Call callback with both raw bytes and formatted string
                                    progress_callback(downloaded_bytes, formatted_progress)
                                    
                                    # Debug: Log the calculation for troubleshooting
                                    logger.debug(f"Progress calc: {percentage}% of {total_size_bytes} bytes = {downloaded_bytes} bytes")
                                except Exception as parse_error:
                                    logger.debug(f"Failed to parse progress line: {line}, error: {parse_error}")
                                    # Fallback: try to extract any percentage from the line
                                    import re
                                    percent_match = re.search(r'([\d.]+)%', line)
                                    if percent_match:
                                        try:
                                            percent = float(percent_match.group(1))
                                            # Estimate bytes based on percentage
                                            if estimated_size_mb:
                                                estimated_bytes = int(estimated_size_mb * 1024 * 1024 * percent / 100)
                                            else:
                                                estimated_bytes = int(percent * 1024 * 1024)  # Assume 1GB
                                            progress_callback(estimated_bytes, line)
                                        except:
                                            progress_callback(0, line)
                                    else:
                                        progress_callback(0, line)
                            else:
                                # For other download messages, just pass them through
                                if line.startswith('[download]'):
                                    progress_callback(0, line)
                        
                        # Also look for completion messages
                        elif 'has already been downloaded' in line or '100%' in line:
                            progress_callback(0, line)
                    
                except Exception as e:
                    logger.error(f"Progress monitoring error: {e}")
                finally:
                    # Close stdout when done
                    try:
                        if process.stdout:
                            process.stdout.close()
                    except:
                        pass
            
            progress_thread = threading.Thread(target=monitor_progress, daemon=True)
            progress_thread.start()
        
        # Wait for completion
        try:
            # Check for cancellation during wait
            if _download_cancelled:
                logger.info("Download cancelled during wait")
                # Clean up and return cancellation
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except:
                    pass
                return ("", None)
            
            # Don't use communicate() since we're already reading stdout in the progress thread
            process.wait()
            
            # Check for cancellation immediately after wait
            if _download_cancelled:
                logger.info("Download cancelled after process completion")
                # Clean up and return cancellation
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except:
                    pass
                return ("", None)
                
        except Exception as e:
            logger.error(f"Error waiting for process completion: {e}")
            # Check if this was a cancellation
            if _download_cancelled:
                logger.info("Download cancelled during wait exception")
                return ("", None)
            raise Exception(f"Download process error: {e}")
        
        if process.returncode != 0:
            # IMMEDIATELY check for cancellation before any other processing
            if _download_cancelled:
                logger.info("Download was cancelled by user - returning early")
                # Clean up and return early without error
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except:
                    pass
                return ("", None)  # Return empty string to indicate cancellation
            
            # Also check if the process was terminated recently (within last 2 seconds)
            import time as time_module
            if hasattr(process, '_termination_time') and time_module.time() - process._termination_time < 2:
                logger.info("Process was recently terminated - treating as cancellation")
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except:
                    pass
                return ("", None)
            
            # Get any remaining stderr output
            try:
                stderr_output = process.stderr.read() if process.stderr else ""
            except:
                stderr_output = ""
            
            error_msg = stderr_output.strip() if stderr_output else "Unknown download error"
            
            # Check if this is a "file busy" error during rename, which is normal when playing
            # We need to check the raw error_msg before any formatting
            raw_error = error_msg  # Keep the raw error for detection
            
            # Check for various file busy error patterns
            has_rename_error = "Unable to rename file" in raw_error
            has_winerror_32 = "WinError 32" in raw_error
            has_giving_up = "Giving up after 3 retries" in raw_error
            
            # Check for file busy error patterns
            is_file_busy_error = (
                has_rename_error and (
                    has_winerror_32 or 
                    "process cannot access the file" in raw_error.lower()
                )
            )
            
            # Also check if we're in a "Giving up after 3 retries" scenario with file rename
            if not is_file_busy_error and has_giving_up and has_rename_error:
                is_file_busy_error = has_winerror_32
            
            if is_file_busy_error:
                logger.info("Download completed but file rename failed due to file being accessed by player. This is normal when watching while downloading.")
                # Check if the final file exists, if not, return the .part file path
                if temp_file.exists():
                    return (str(temp_file), video_title)
                else:
                    # The file was not renamed, return the .part file path
                    part_file = Path(str(temp_file) + ".part")
                    if part_file.exists():
                        logger.info(f"Returning .part file since final file doesn't exist: {part_file}")
                        return (str(part_file), video_title)
                    else:
                        # Fallback to any video file in the directory
                        temp_dir = temp_file.parent
                        for video_ext in ['*.mp4', '*.webm', '*.mkv', '*.avi', '*.mov', '*.m4v']:
                            for video_file in temp_dir.glob(video_ext):
                                if video_file.stat().st_size > 1024*1024:  # At least 1MB
                                    logger.info(f"Returning existing video file: {video_file}")
                                    return (str(video_file), video_title)
                        # If no video file found, return the expected path anyway
                        return (str(temp_file), video_title)
            
            # Provide specific error messages
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise Exception("Rate limit exceeded. Please wait a few minutes and try again.")
            elif "signature" in error_msg.lower():
                raise Exception("Video signature verification failed. Please try again later.")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise Exception("Network error occurred during download. Please check your internet connection and try again.")
            else:
                raise Exception(f"Download failed: {error_msg}")
        
        # Check if file exists and is readable
        if not temp_file.exists():
            raise Exception("Download completed but no file was created")
        
        # Final progress update with error handling
        if progress_callback:
            try:
                final_size = temp_file.stat().st_size
                progress_callback(final_size)
            except Exception as e:
                pass  # Silent error handling
        
        logger.info(f"Progressive download completed: {temp_file}")
        return (str(temp_file), video_title)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp command failed: {e}")
        raise Exception(f"yt-dlp error: {e.stderr or str(e)}")
    except Exception as e:
        logger.error(f"Progressive download failed: {e}")
        # Clean up on failure
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass
        raise