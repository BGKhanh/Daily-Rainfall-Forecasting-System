# -*- coding: utf-8 -*-
"""Pytest root configuration and early bootstrap isolation guards.

Activates PreReadFileReadGuard during pytest session configuration before test collection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.protocols import get_global_read_guard


def pytest_configure(config):
    """Activate fail-closed PreReadFileReadGuard prior to test collection and execution."""
    guard = get_global_read_guard()
    guard.activate()

