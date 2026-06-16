# -*- coding: utf-8 -*-
"""
Pytest configuration: set a non-interactive matplotlib backend and disable
LaTeX rendering so plot tests pass on CI runners that have no LaTeX install.
"""

import matplotlib

matplotlib.use("Agg")  # must come before any pyplot import

import pytest
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _no_latex():
    """Force text.usetex=False for every test regardless of the loaded style."""
    with plt.rc_context({"text.usetex": False}):
        yield
