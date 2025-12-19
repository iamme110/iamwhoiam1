# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
File utilities for download operations
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


class DownloadFileManager:
    """Manages temporary files for downloads"""

    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def get_temp_path(self, filename: str) -> str:
        """Get a temporary path for downloading"""
        return str(self.temp_dir / filename)

    def cleanup_temp_file(self, file_path: str) -> None:
        """Remove a temporary file if it exists"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # Ignore cleanup errors

    def get_file_size(self, file_path: str) -> int:
        """Get file size, return 0 if file doesn't exist"""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    def validate_download(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Validate downloaded file"""
        if not os.path.exists(file_path):
            return False, "Downloaded file does not exist"

        size = self.get_file_size(file_path)
        if size == 0:
            self.cleanup_temp_file(file_path)
            return False, "Downloaded file is empty"

        return True, None


def extract_filename_from_url(url: str) -> str:
    """Extract filename from URL"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    return filename or 'downloaded_video.mp4'


def extract_filename_from_headers(response) -> Optional[str]:
    """Extract filename from Content-Disposition header"""
    if 'Content-Disposition' in response.headers:
        import re
        match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', response.headers['Content-Disposition'])
        if match:
            return match.group(1).strip('\'"')
    return None


def guess_extension_from_content_type(content_type: str) -> str:
    """Guess file extension from content type"""
    import mimetypes
    ext = mimetypes.guess_extension(content_type)
    return ext or '.mp4'


def generate_filename(url: str, response=None) -> str:
    """Generate appropriate filename for download"""
    filename = None

    # Try Content-Disposition header first
    if response:
        filename = extract_filename_from_headers(response)

    # Fall back to URL
    if not filename:
        filename = extract_filename_from_url(url)

    # Ensure it has an extension
    if response and (not filename or '.' not in filename):
        content_type = response.headers.get('Content-Type', 'video/mp4')
        ext = guess_extension_from_content_type(content_type)
        if not filename:
            filename = f'downloaded_video{ext}'
        else:
            filename = f"{filename}{ext}"

    return filename
    """Generate appropriate filename for download"""
    filename = None

    # Try Content-Disposition header first
    if response:
        filename = extract_filename_from_headers(response)

    # Fall back to URL
    if not filename:
        filename = extract_filename_from_url(url)

    # Ensure it has an extension
    if response and not filename or '.' not in filename:
        content_type = response.headers.get('Content-Type', 'video/mp4')
        ext = guess_extension_from_content_type(content_type)
        if not filename:
            filename = f'downloaded_video{ext}'
        else:
            filename = f"{filename}{ext}"

    return filename