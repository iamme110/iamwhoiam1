# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
URL Handler Factory for different protocols
"""

from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urlparse


class URLHandler(ABC):
    """Base class for URL handlers"""

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this handler can handle the URL"""
        pass

    @abstractmethod
    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL and return (is_valid, error_message)"""
        pass

    @abstractmethod
    def get_filename(self, url: str) -> str:
        """Extract filename from URL"""
        pass


class WebURLHandler(URLHandler):
    """Handler for HTTP/HTTPS URLs"""

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme.lower() in ('http', 'https')

    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        # Basic validation - could be extended with HEAD requests
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "Invalid URL format"
        return True, None

    def get_filename(self, url: str) -> str:
        return url.split('/')[-1] or 'downloaded_video.mp4'


class FTPURLHandler(URLHandler):
    """Handler for FTP/FTPS URLs"""

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme.lower() in ('ftp', 'ftps')

    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "Invalid FTP URL format"
        return True, None

    def get_filename(self, url: str) -> str:
        import os
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        return filename or 'downloaded_video.mp4'


class FileURLHandler(URLHandler):
    """Handler for file:// URLs"""

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme.lower() in ('file', 'smb')

    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        import os
        import urllib.parse

        parsed = urlparse(url)
        if parsed.scheme == 'file':
            file_path = urllib.parse.unquote(parsed.path)
            if os.name == 'nt' and file_path.startswith('/'):
                file_path = file_path[1:]
            if os.path.exists(file_path):
                return True, None
            else:
                return False, "File does not exist"
        return False, "Unsupported file URL scheme"

    def get_filename(self, url: str) -> str:
        import os
        import urllib.parse

        parsed = urlparse(url)
        file_path = urllib.parse.unquote(parsed.path)
        if os.name == 'nt' and file_path.startswith('/'):
            file_path = file_path[1:]
        return os.path.basename(file_path)


class SMBPathHandler(URLHandler):
    r"""Handler for Windows SMB paths (\\server\share\path)"""

    def can_handle(self, url: str) -> bool:
        return url.startswith('\\\\')

    def validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        import os
        # Convert to file path and check
        file_path = url.replace('\\', '/')
        if os.path.exists(file_path):
            return True, None
        else:
            return False, "SMB path does not exist or is not accessible"

    def get_filename(self, url: str) -> str:
        import os
        return os.path.basename(url.replace('\\', '/'))


class URLHandlerFactory:
    """Factory for creating URL handlers"""

    _handlers = [
        WebURLHandler(),
        FTPURLHandler(),
        FileURLHandler(),
        SMBPathHandler(),
    ]

    @classmethod
    def get_handler(cls, url: str) -> Optional[URLHandler]:
        """Get appropriate handler for URL"""
        for handler in cls._handlers:
            if handler.can_handle(url):
                return handler
        return None

    @classmethod
    def validate_url(cls, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL using appropriate handler"""
        handler = cls.get_handler(url)
        if handler:
            return handler.validate_url(url)
        return False, f"No handler available for URL scheme: {urlparse(url).scheme}"

    @classmethod
    def get_filename(cls, url: str) -> str:
        """Get filename using appropriate handler"""
        handler = cls.get_handler(url)
        if handler:
            return handler.get_filename(url)
        return 'downloaded_video.mp4'