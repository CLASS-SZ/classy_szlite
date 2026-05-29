"""CI-friendly smoke tests for classy_szlite.likelihoods.

These exercise import-time code paths and the data-loader's error reporting
without needing the heavy likelihood data files to be present. The full
chi2/grad tests live in test_likelihoods.py and run only when the npz
tables are installed.
"""
from __future__ import annotations
import os
from pathlib import Path
import pytest


def test_likelihoods_public_api():
    """Public functions are importable and look like callables."""
    from classy_szlite.likelihoods import (
        chi2_lowTT, chi2_sroll2, chi2_plac,
        chi2_mflike, chi2_mflike_v2,
        fg_totals_jax, total_chi2,
    )
    for fn in (chi2_lowTT, chi2_sroll2, chi2_plac,
                chi2_mflike, chi2_mflike_v2,
                fg_totals_jax, total_chi2):
        assert callable(fn)


def test_cl_TTTEEE_jax_in_top_level():
    import classy_szlite as csl
    assert callable(csl.cl_TTTEEE_jax)
    # The two ell_convention strings round-trip identically on the emulator
    # output (Cl × ell² invariant under shift). This already runs in CI
    # because the emulator weights are downloaded by the test workflow.
    out_a = csl.cl_TTTEEE_jax(csl.CosmoParams(), ell_factor=False,
                                ell_convention="classy_szfast")
    out_b = csl.cl_TTTEEE_jax(csl.CosmoParams(), ell_factor=False,
                                ell_convention="emulator_modes")
    import numpy as np
    np.testing.assert_allclose(
        out_a["tt"] * out_a["ell"] ** 2,
        out_b["tt"] * out_b["ell"] ** 2,
        rtol=1e-10,
    )


def test_data_loader_error_message(tmp_path, monkeypatch):
    """When no data file exists, the loader reports every candidate path."""
    # Point everything at an empty tmpdir so all candidates miss.
    monkeypatch.setenv("CLASSY_SZLITE_LIKELIHOOD_DATA",
                       str(tmp_path / "definitely_not_here.npz"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    from classy_szlite.likelihoods import _data
    _data.load.cache_clear()
    _data.load_fg_components.cache_clear()
    with pytest.raises(FileNotFoundError) as exc_info:
        _data.load()
    msg = str(exc_info.value)
    assert "CLASSY_SZLITE_LIKELIHOOD_DATA" in msg
    assert "extract_data" in msg
    # Cleanup so subsequent tests in the same process see a clean cache.
    _data.load.cache_clear()


def test_data_candidate_paths_order(monkeypatch, tmp_path):
    """The env var takes precedence over the home-dir default."""
    monkeypatch.setenv("CLASSY_SZLITE_LIKELIHOOD_DATA", "/tmp/explicit.npz")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from classy_szlite.likelihoods import _data
    cands = _data._candidate_paths()
    assert cands[0] == Path("/tmp/explicit.npz")
    assert any(".classy_szlite" in str(p) for p in cands)
    assert any("class_sz_data" in str(p) for p in cands)


def test_load_fg_components_warns_when_missing(monkeypatch, tmp_path):
    """Missing fg-components file produces a RuntimeWarning, not an error."""
    monkeypatch.setenv("CLASSY_SZLITE_LIKELIHOOD_DATA",
                       str(tmp_path / "no_such_npz.npz"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from classy_szlite.likelihoods import _data
    _data.load.cache_clear()
    _data.load_fg_components.cache_clear()
    with pytest.warns(RuntimeWarning, match="mflike_fg_components.npz"):
        result = _data.load_fg_components()
    assert result is None
    _data.load_fg_components.cache_clear()


def test_jnp_table_round_trips_endianness():
    """jnp_table converts non-native byte order to native float64."""
    import numpy as np
    from classy_szlite.likelihoods._data import jnp_table
    # Big-endian f32 → native f64 jnp array
    arr = np.asarray([1.0, 2.0, 3.0], dtype=">f4")
    out = jnp_table(arr)
    np.testing.assert_allclose(np.asarray(out), [1.0, 2.0, 3.0])
    assert out.dtype.name == "float64"


def test_total_chi2_raises_when_data_missing(monkeypatch, tmp_path):
    """The CI-installable end-to-end path raises a clear FileNotFoundError
    rather than something cryptic when the data tables aren't there."""
    monkeypatch.setenv("CLASSY_SZLITE_LIKELIHOOD_DATA",
                       str(tmp_path / "missing.npz"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from classy_szlite.likelihoods import _data, core, foreground
    _data.load.cache_clear()
    _data.load_fg_components.cache_clear()
    # Reset module-level caches so the new env var takes effect.
    core._TABLES.clear()
    foreground._TABLES = None
    import classy_szlite as csl
    cosmo = csl.CosmoParams()
    with pytest.raises((FileNotFoundError, KeyError)):
        # Calling with no params and missing data should fail fast.
        from classy_szlite.likelihoods import total_chi2
        total_chi2(cosmo)
    _data.load.cache_clear()
    _data.load_fg_components.cache_clear()
    core._TABLES.clear()
    foreground._TABLES = None
