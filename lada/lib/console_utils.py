# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import os


def hide_console_window():
    """
    Hide the console window on supported platforms.
    Currently only supports Windows.
    """
    if os.name == 'nt':
        try:
            import ctypes
            console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if console_hwnd:
                ctypes.windll.user32.ShowWindow(console_hwnd, 0)  # 0 = SW_HIDE
        except (AttributeError, OSError):
            # Console hiding failed, continue without hiding
            pass


def show_console_window():
    if os.name == 'nt':
        try:
            import ctypes
            console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            
            # If no console window exists (e.g., PyInstaller console=False), allocate one
            if not console_hwnd:
                # Try to attach to parent's console first
                if not ctypes.windll.kernel32.AttachConsole(ctypes.windll.kernel32.ATTACH_PARENT_PROCESS):
                    # If no parent console, allocate a new one
                    ctypes.windll.kernel32.AllocConsole()
                console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            
            if console_hwnd:
                ctypes.windll.user32.ShowWindow(console_hwnd, 1)  # 1 = SW_SHOWNORMAL
        except (AttributeError, OSError):
            # Console showing failed, continue without showing
            pass
