"""Command-line entry point for extracting the JAX-likelihood data tables.

Run on a machine where ``cobaya`` + the relevant likelihood data packages are
installed (and a working ``act_dr6_cmbonly.PlanckActCut`` is on the path):

.. code-block:: bash

    python -m classy_szlite.likelihoods.extract_data \\
        --chain-yaml /path/to/p-actbase_ede+n3_classsz.input.yaml \\
        --packages-path ~/cobaya_packages \\
        --out ~/.classy_szlite/

This produces two npz files in ``--out``:

  * ``likelihood_data.npz`` (~150 MB) — bandpowers, covariances, window
    functions and fixed-foreground totals for the four likelihoods.
  * ``mflike_fg_components.npz`` (~25 MB) — per-component foreground
    templates used by :func:`chi2_mflike_v2`.

Subsequently, point :envvar:`CLASSY_SZLITE_LIKELIHOOD_DATA` at the main npz
(the foreground components are picked up from the same directory):

.. code-block:: bash

    export CLASSY_SZLITE_LIKELIHOOD_DATA=~/.classy_szlite/likelihood_data.npz
"""
from __future__ import annotations
import argparse
import os
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m classy_szlite.likelihoods.extract_data")
    p.add_argument("--chain-yaml", type=Path, default=None,
                    help="Path to the cobaya .input.yaml of the chain to extract.")
    p.add_argument("--packages-path", type=Path, default=None,
                    help="Cobaya packages_path (where likelihood data lives).")
    p.add_argument("--out", type=Path, default=Path.home() / ".classy_szlite",
                    help="Output directory (default: ~/.classy_szlite).")
    p.add_argument("--skip-fg", action="store_true",
                    help="Skip the per-component foreground extraction.")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    env_overrides = {}
    if args.chain_yaml is not None:
        env_overrides["CLASSY_SZLITE_LIK_CHAIN_YAML"] = str(args.chain_yaml)
    if args.packages_path is not None:
        env_overrides["CLASSY_SZLITE_LIK_PACKAGES_PATH"] = str(args.packages_path)
    env_overrides["CLASSY_SZLITE_LIK_OUT_NPZ"] = str(args.out / "likelihood_data.npz")
    env_overrides["CLASSY_SZLITE_LIK_FG_NPZ"]  = str(args.out / "mflike_fg_components.npz")
    os.environ.update(env_overrides)

    print(f"=== Stage 1: bandpowers / covmats / window functions ===")
    runpy.run_module("classy_szlite.likelihoods._extract_main", run_name="__main__")

    if not args.skip_fg:
        print(f"\n=== Stage 2: per-component foreground templates ===")
        runpy.run_module("classy_szlite.likelihoods._extract_fg", run_name="__main__")

    print(f"\nDone. Set:\n"
          f"    export CLASSY_SZLITE_LIKELIHOOD_DATA={args.out / 'likelihood_data.npz'}\n"
          f"to pick this data up automatically.")


if __name__ == "__main__":
    main()
