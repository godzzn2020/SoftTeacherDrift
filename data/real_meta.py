"""真实数据集的元信息辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

DEFAULT_INSECTS_META = {
    "name": "INSECTS_abrupt_balanced",
    "path": "datasets/real/INSECTS_abrupt_balanced.csv",
    "indexing": "0-based",
    "drift_type": "abrupt",
    "positions": [14351, 19499, 33239, 38681, 39509],
}


def load_insects_abrupt_balanced_meta(
    meta_path: str = "datasets/real/INSECTS_abrupt_balanced.json",
) -> Dict[str, object]:
    """
    加载 INSECTS_abrupt_balanced 的元数据，如无文件则写入默认模板。
    假设 positions 字段为 0-based 漂移起点索引。
    """
    path = Path(meta_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_INSECTS_META, f, indent=2)
    return json.loads(path.read_text(encoding="utf-8"))
