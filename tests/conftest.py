# -*- coding: utf-8 -*-
"""
Pytest configuration: non-interactive matplotlib backend for CI.

LaTeX fallback is handled in the source (plot.py and ks_analysis.py) by
checking shutil.which("latex") after every style.use() call.  This fixture
is a safety net that disables text.usetex before and after each test.
"""

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _no_latex():
    """Guarantee text.usetex=False throughout every test."""
    matplotlib.rcParams["text.usetex"] = False
    yield
    matplotlib.rcParams["text.usetex"] = False
