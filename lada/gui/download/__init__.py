# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
URL Download Module for Lada GUI

This module provides comprehensive URL download functionality
with support for multiple protocols, authentication, and progress tracking.
"""

from .downloader import URLDownloader
from .url_handler import URLHandlerFactory
from .progress_dialog import ProgressDialog
from .auth_dialog import AuthDialog

__all__ = [
    'URLDownloader',
    'URLHandlerFactory',
    'ProgressDialog',
    'AuthDialog'
]