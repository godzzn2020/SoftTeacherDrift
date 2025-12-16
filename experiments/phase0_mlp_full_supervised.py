"""Compat 脚本：沿用 phase0_offline_supervised 的实现，以 `phase0_mlp_full_supervised.py` 名称暴露给调度器。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.phase0_offline_supervised import main as run_phase0_offline


def main() -> None:
    run_phase0_offline()


if __name__ == "__main__":
    main()
