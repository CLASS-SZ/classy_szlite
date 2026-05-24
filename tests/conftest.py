"""Shared test fixtures + emulator-data availability gate."""
from __future__ import annotations
import pytest

try:
    import classy_szlite  # noqa: F401
    from classy_szlite._emulator import default_data_dir
    default_data_dir()                                # raises if data missing
    DATA_OK = True
except Exception:
    DATA_OK = False


def pytest_collection_modifyitems(config, items):
    """Skip every test if emulator data is unavailable (so CI passes)."""
    if DATA_OK:
        return
    skip = pytest.mark.skip(reason="classy_szlite emulator data not available")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def cosmo():
    """Default ede-v2 LCDM-equivalent cosmology."""
    import classy_szlite as csl
    return csl.CosmoParams()


@pytest.fixture(scope="session")
def profile():
    """Reference Arnaud-10 profile (matches may26 fionapaper baseline)."""
    import classy_szlite as csl
    return csl.ProfileParamsA10(P0=8.13, beta=5.48, B=1.25)
