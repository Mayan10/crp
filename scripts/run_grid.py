"""Run the experiment grid in order, skipping runs that already finished.

Safe to stop and restart at any point: a run whose test metrics already exist
is skipped, and a run that was interrupted part way resumes from its last
completed round.

Usage:
    python scripts/run_grid.py --group core --dry-run
    python scripts/run_grid.py --group core
    python scripts/run_grid.py --group mu_selection
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config


@dataclass
class Job:
    """One run: a config file plus the overrides that distinguish it."""

    label: str
    script: str
    config: str
    overrides: list[str]

    def command(self, extra: list[str]) -> list[str]:
        cmd = [sys.executable, "-u", self.script, "--config", self.config]
        if self.overrides or extra:
            cmd += ["--set", *self.overrides, *extra]
        return cmd


def centralized_jobs(seeds: list[int]) -> list[Job]:
    return [
        Job(f"centralized seed{s}", "scripts/train_centralized.py",
            "configs/centralized.yaml", [f"seed={s}"])
        for s in seeds
    ]


def fedavg_iid_jobs(seeds: list[int]) -> list[Job]:
    return [
        Job(f"fedavg iid seed{s}", "scripts/run_federated.py",
            "configs/fedavg_iid.yaml", [f"seed={s}", f"partition.seed={s}"])
        for s in seeds
    ]


def fedavg_noniid_jobs(alphas: list[float], seeds: list[int]) -> list[Job]:
    return [
        Job(f"fedavg alpha{a:g} seed{s}", "scripts/run_federated.py",
            "configs/fedavg_noniid.yaml",
            [f"seed={s}", f"partition.seed={s}", f"partition.alpha={a}"])
        for a in alphas for s in seeds
    ]


def mu_selection_jobs(mus: list[float], alpha: float = 0.1, seed: int = 0) -> list[Job]:
    """mu is chosen on validation accuracy at the hardest alpha, seed 0 only."""
    return [
        Job(f"fedprox mu{m:g} alpha{alpha:g} seed{seed}", "scripts/run_federated.py",
            "configs/fedprox_noniid.yaml",
            [f"seed={seed}", f"partition.seed={seed}", f"partition.alpha={alpha}",
             f"federated.mu={m}"])
        for m in mus
    ]


def fedprox_jobs(mu: float, alphas: list[float], seeds: list[int]) -> list[Job]:
    return [
        Job(f"fedprox mu{mu:g} alpha{a:g} seed{s}", "scripts/run_federated.py",
            "configs/fedprox_noniid.yaml",
            [f"seed={s}", f"partition.seed={s}", f"partition.alpha={a}", f"federated.mu={mu}"])
        for a in alphas for s in seeds
    ]


def legacy_job() -> list[Job]:
    return [
        Job("legacy reproduction", "scripts/run_federated.py",
            "configs/legacy_reproduction.yaml", ["seed=0", "partition.seed=0"])
    ]


GROUPS = {
    "centralized": lambda a: centralized_jobs(a.seeds),
    "fedavg_iid": lambda a: fedavg_iid_jobs(a.seeds),
    "fedavg_noniid": lambda a: fedavg_noniid_jobs(a.alphas, a.seeds),
    "mu_selection": lambda a: mu_selection_jobs(a.mus),
    "fedprox_noniid": lambda a: fedprox_jobs(a.mu, a.alphas, a.seeds),
    "legacy": lambda a: legacy_job(),
}
CORE_ORDER = ["centralized", "fedavg_iid", "fedavg_noniid", "mu_selection", "fedprox_noniid"]


def build_jobs(args) -> list[Job]:
    if args.group == "core":
        jobs: list[Job] = []
        for group in CORE_ORDER:
            jobs += GROUPS[group](args)
        return jobs
    if args.group == "all":
        return build_jobs_for(["core", "legacy"], args)
    return GROUPS[args.group](args)


def build_jobs_for(groups: list[str], args) -> list[Job]:
    jobs: list[Job] = []
    for group in groups:
        if group == "core":
            for sub in CORE_ORDER:
                jobs += GROUPS[sub](args)
        else:
            jobs += GROUPS[group](args)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="core",
                        choices=["core", "all", *GROUPS.keys()])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--mu", type=float, default=0.01,
                        help="mu for the fedprox_noniid group, set from the selection result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="extra overrides for every job")
    args = parser.parse_args()

    jobs = build_jobs(args)
    print(f"{len(jobs)} runs in group {args.group!r}\n")

    for i, job in enumerate(jobs, 1):
        command = job.command(args.set) + ["--skip-existing"]
        print(f"[{i}/{len(jobs)}] {job.label}")
        if args.dry_run:
            print(f"    {' '.join(command[1:])}")
            continue

        started = time.time()
        result = subprocess.run(command, cwd=REPO)
        elapsed = time.time() - started

        if result.returncode != 0:
            print(f"    FAILED after {elapsed / 60:.1f} min (exit {result.returncode})")
            if args.stop_on_failure:
                return result.returncode
        else:
            print(f"    done in {elapsed / 60:.1f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
