# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from dataclasses import dataclass


@dataclass
class AuthCredentials:
    """Authentication credentials"""
    username: str
    password: str