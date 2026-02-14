# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO Dashboard Module

Web-based interactive dashboard for real-time visualization and analysis.
"""

from .app import app, create_dashboard

__all__ = ['app', 'create_dashboard']
