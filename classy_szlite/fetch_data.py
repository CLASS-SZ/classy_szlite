"""Download the ede-v2 CosmoPower emulator data for classy_szlite.

The emulator weights (~115 MB) live in a separate repository,
``cosmopower-organization/ede``, and are not bundled in the PyPI wheel.
This module fetches exactly the pickle-free ``*_v2_plain.npz`` files that
:mod:`classy_szlite` needs, places them under the data directory, and
verifies each one loads without pickling.

Usage
-----
Command line (installed as a console script too)::

    python -m classy_szlite.fetch_data
    classy-szlite-get-data            # equivalent console-script entry point

Options::

    --dest DIR     where to put the data (default: $CLASSY_SZLITE_DATA_DIR,
                   else ~/class_sz_data). The files land under <DIR>/ede/...
    --force        re-download even if a valid file is already present
    --quiet        only print warnings/errors

After it finishes, classy_szlite picks the data up automatically (it checks
``$CLASSY_SZLITE_DATA_DIR``, then ``~/class_sz_data``, then the legacy
``~/class_sz_data_directory``). If you used a custom ``--dest`` that is not
one of those, export it::

    export CLASSY_SZLITE_DATA_DIR=/your/dest
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ._registry import _DATA_SUBDIR, _EMULATOR_FILES

# Raw base URL for the pickle-free emulator files.
_BASE_URL = "https://raw.githubusercontent.com/cosmopower-organization/ede/main"


def _default_dest() -> Path:
    """Resolve where to download to (mirrors `_emulator.default_data_dir`,
    but does not require the directory to already exist)."""
    env = os.environ.get("CLASSY_SZLITE_DATA_DIR")
    if env:
        return Path(env).expanduser()
    # Prefer an existing recognised location; otherwise default to the
    # recommended ~/class_sz_data.
    for candidate in ("~/class_sz_data", "~/class_sz_data_directory"):
        p = Path(candidate).expanduser()
        if p.is_dir():
            return p
    return Path("~/class_sz_data").expanduser()


def _loads_pickle_free(path: Path) -> bool:
    """True iff the npz exists and every member loads with allow_pickle=False."""
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as d:
            for k in d.files:
                _ = d[k]               # force-read every member
        return True
    except Exception:
        return False


def _download_one(rel: str, dest_root: Path, *, force: bool, quiet: bool) -> str:
    """Download a single emulator file. Returns 'ok' | 'skip' | 'fail:<msg>'."""
    out = dest_root / _DATA_SUBDIR / rel
    if out.is_file() and not force and _loads_pickle_free(out):
        if not quiet:
            print(f"  skip  {rel}  (already present and valid)")
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE_URL}/{rel}"
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        # Stream to a .part file so a failed/partial download never replaces a
        # good file.
        with urllib.request.urlopen(url, timeout=60) as resp:
            if getattr(resp, "status", 200) not in (200, None):
                return f"fail:HTTP {resp.status}"
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)   # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        return f"fail:HTTP {e.code}"
    except Exception as e:                                       # network, timeout…
        tmp.unlink(missing_ok=True)
        return f"fail:{type(e).__name__}: {e}"

    # Validate before committing the file into place.
    if not _loads_pickle_free(tmp):
        tmp.unlink(missing_ok=True)
        return ("fail:downloaded file is not a pickle-free npz "
                "(wrong URL or corrupted download)")
    tmp.replace(out)
    if not quiet:
        size_mb = out.stat().st_size / 1e6
        print(f"  ok    {rel}  ({size_mb:.1f} MB)")
    return "ok"


def fetch(dest: str | os.PathLike | None = None, *,
          force: bool = False, quiet: bool = False) -> Path:
    """Download all required emulator files. Returns the destination root.

    Raises ``RuntimeError`` if any file fails to download/validate.
    """
    dest_root = Path(dest).expanduser() if dest else _default_dest()
    if not quiet:
        print(f"classy_szlite: fetching {len(_EMULATOR_FILES)} ede-v2 emulator "
              f"files into {dest_root / _DATA_SUBDIR}")
    failures = []
    for rel in _EMULATOR_FILES.values():
        status = _download_one(rel, dest_root, force=force, quiet=quiet)
        if status.startswith("fail"):
            print(f"  FAIL  {rel}  — {status[5:]}", file=sys.stderr)
            failures.append(rel)
    if failures:
        raise RuntimeError(
            f"classy_szlite.fetch_data: {len(failures)} file(s) failed: "
            + ", ".join(failures)
        )
    if not quiet:
        print(f"\nDone. classy_szlite will use {dest_root / _DATA_SUBDIR}.")
        env = os.environ.get("CLASSY_SZLITE_DATA_DIR")
        recognised = {Path("~/class_sz_data").expanduser(),
                      Path("~/class_sz_data_directory").expanduser()}
        if not env and dest_root not in recognised:
            print(f"NOTE: {dest_root} is not an auto-detected location; add\n"
                  f"      export CLASSY_SZLITE_DATA_DIR={dest_root}\n"
                  f"      to your shell profile so classy_szlite finds it.")
    return dest_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m classy_szlite.fetch_data",
        description="Download the ede-v2 CosmoPower emulator data for classy_szlite.",
    )
    p.add_argument("--dest", default=None,
                   help="destination dir (default: $CLASSY_SZLITE_DATA_DIR, "
                        "else ~/class_sz_data). Files go under <dest>/ede/.")
    p.add_argument("--force", action="store_true",
                   help="re-download even if a valid file is already present.")
    p.add_argument("--quiet", action="store_true",
                   help="only print warnings and errors.")
    args = p.parse_args(argv)
    try:
        fetch(args.dest, force=args.force, quiet=args.quiet)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
