"""Network-free tests for classy_szlite.fetch_data.

We monkeypatch ``urllib.request.urlopen`` to serve in-memory bytes, so the
download/validate/skip/repair logic is exercised without touching the
network.
"""
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import pytest

from classy_szlite import fetch_data


# ── helpers to build valid / invalid npz blobs ────────────────────────────
def _good_npz_bytes() -> bytes:
    """A pickle-free npz that loads with allow_pickle=False."""
    buf = io.BytesIO()
    np.savez(buf, **{"weights_.n": np.array(0), "a": np.arange(3.0)})
    return buf.getvalue()


def _bad_npz_bytes() -> bytes:
    """A pickled npz (object array) — the failure mode we guard against."""
    buf = io.BytesIO()
    np.savez(buf, arr_0=np.array({"x": 1}, dtype=object))
    return buf.getvalue()


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


@pytest.fixture
def serve(monkeypatch):
    """Return a setter that makes urlopen serve given bytes for every URL."""
    state = {"payload": _good_npz_bytes(), "calls": 0}

    def fake_urlopen(url, timeout=None):
        state["calls"] += 1
        return _FakeResponse(state["payload"])

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", fake_urlopen)
    return state


def test_fetch_all_good(tmp_path, serve):
    fetch_data.fetch(dest=tmp_path, quiet=True)
    # All 10 files present and valid.
    from classy_szlite._registry import _EMULATOR_FILES, _DATA_SUBDIR
    for rel in _EMULATOR_FILES.values():
        p = tmp_path / _DATA_SUBDIR / rel
        assert p.is_file()
        assert fetch_data._loads_pickle_free(p)
    assert serve["calls"] == len(_EMULATOR_FILES)


def test_fetch_is_idempotent(tmp_path, serve):
    fetch_data.fetch(dest=tmp_path, quiet=True)
    first = serve["calls"]
    # Second run should re-validate and skip — no new downloads.
    fetch_data.fetch(dest=tmp_path, quiet=True)
    assert serve["calls"] == first


def test_invalid_existing_file_is_repaired(tmp_path, serve):
    from classy_szlite._registry import _EMULATOR_FILES, _DATA_SUBDIR
    fetch_data.fetch(dest=tmp_path, quiet=True)
    # Corrupt one file with a pickled blob (Fiona's exact scenario).
    rel = next(iter(_EMULATOR_FILES.values()))
    victim = tmp_path / _DATA_SUBDIR / rel
    victim.write_bytes(_bad_npz_bytes())
    assert not fetch_data._loads_pickle_free(victim)
    # Re-run WITHOUT --force should still repair it.
    before = serve["calls"]
    fetch_data.fetch(dest=tmp_path, quiet=True)
    assert serve["calls"] == before + 1          # exactly the one re-download
    assert fetch_data._loads_pickle_free(victim)


def test_bad_download_does_not_replace_good_file(tmp_path, serve):
    from classy_szlite._registry import _EMULATOR_FILES, _DATA_SUBDIR
    fetch_data.fetch(dest=tmp_path, quiet=True)
    # Now make the server return garbage and force a re-download.
    serve["payload"] = _bad_npz_bytes()
    with pytest.raises(RuntimeError, match="failed"):
        fetch_data.fetch(dest=tmp_path, force=True, quiet=True)
    # The previously-good files must survive (download validated before commit).
    for rel in _EMULATOR_FILES.values():
        p = tmp_path / _DATA_SUBDIR / rel
        assert p.is_file() and fetch_data._loads_pickle_free(p)
        # no leftover .part temp files
        assert not p.with_suffix(p.suffix + ".part").exists()


def test_default_dest_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLASSY_SZLITE_DATA_DIR", str(tmp_path / "envdir"))
    assert fetch_data._default_dest() == tmp_path / "envdir"


def test_main_returns_zero_on_success(tmp_path, serve):
    rc = fetch_data.main(["--dest", str(tmp_path), "--quiet"])
    assert rc == 0


def test_main_returns_one_on_failure(tmp_path, serve):
    serve["payload"] = _bad_npz_bytes()
    rc = fetch_data.main(["--dest", str(tmp_path), "--quiet"])
    assert rc == 1
