"""Select the FedProx mu on validation accuracy, never on test.

Reads the per round validation curves from the mu selection runs and reports
the mu with the highest best validation accuracy. The test split plays no part
in this choice, which is what keeps the later test numbers an estimate of
generalization rather than of the selection.

Usage:
    python scripts/select_mu.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    federated = Path(cfg.results_dir) / "federated"

    rows = []
    for history_path in sorted(federated.glob("fedprox_*/history.csv")):
        summary_path = history_path.parent / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        name = history_path.parent.name
        if f"alpha{args.alpha:g}" not in name or f"seed{args.seed}" not in name:
            continue

        history = pd.read_csv(history_path)
        best = history.loc[history["val_accuracy"].idxmax()]
        rows.append(
            {
                "run": name,
                "mu": summary.get("mu"),
                "best_val_accuracy": float(best["val_accuracy"]),
                "best_val_macro_f1": float(best["val_macro_f1"]),
                "best_round": int(best["round"]),
                "val_correct": int(best["val_n_correct"]),
                "val_total": int(best["val_n_total"]),
            }
        )

    if not rows:
        print(f"no FedProx runs found at alpha {args.alpha:g}, seed {args.seed}. "
              f"Run: python scripts/run_grid.py --group mu_selection")
        return 1

    table = pd.DataFrame(rows).sort_values("best_val_accuracy", ascending=False)
    print(f"FedProx mu selection at alpha {args.alpha:g}, seed {args.seed}, on validation only\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = table.iloc[0]
    print(f"\nselected mu = {best['mu']:g} "
          f"(validation accuracy {best['best_val_accuracy']:.4f} at round {best['best_round']})")

    out_dir = Path(cfg.results_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "mu_selection.csv", index=False)

    with open(out_dir / "mu_selection.json", "w") as handle:
        json.dump(
            {
                "selected_mu": float(best["mu"]),
                "alpha": args.alpha,
                "seed": args.seed,
                "criterion": "highest best validation accuracy, test not consulted",
                "candidates": rows,
            },
            handle,
            indent=2,
        )
    print(f"wrote {out_dir / 'mu_selection.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
