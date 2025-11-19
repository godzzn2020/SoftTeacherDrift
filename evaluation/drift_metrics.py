"""概念漂移检测指标实现。"""

from __future__ import annotations

from typing import Dict, Sequence, List
import math


def compute_detection_metrics(
    gt_drifts: Sequence[int],
    detections: Sequence[int],
    T: int,
) -> Dict[str, float]:
    """
    根据 Lukats 等定义计算 MDR、MTD、MTFA、MTR。
    """
    gt = sorted(int(d) for d in gt_drifts)
    dets = sorted(int(d) for d in detections)
    total_gt = len(gt)
    if total_gt == 0 or T <= 0:
        return {"MDR": math.nan, "MTD": math.nan, "MTFA": math.nan, "MTR": math.nan}
    missed = 0
    delays: List[int] = []
    false_alarms: List[int] = []
    det_idx = 0
    for i, g in enumerate(gt):
        end = gt[i + 1] if i + 1 < total_gt else T
        # false alarms before the current drift window
        while det_idx < len(dets) and dets[det_idx] < g:
            false_alarms.append(dets[det_idx])
            det_idx += 1
        detected = False
        while det_idx < len(dets) and dets[det_idx] < end:
            if not detected:
                delays.append(max(0, dets[det_idx] - g))
                detected = True
            else:
                false_alarms.append(dets[det_idx])
            det_idx += 1
        if not detected:
            missed += 1
    while det_idx < len(dets):
        false_alarms.append(dets[det_idx])
        det_idx += 1

    mdr = missed / total_gt
    mtd = sum(delays) / len(delays) if delays else math.nan
    if len(false_alarms) >= 2:
        gaps = [b - a for a, b in zip(false_alarms[:-1], false_alarms[1:])]
        mtfa = sum(gaps) / len(gaps)
    else:
        mtfa = math.nan
    denominator = (mtd if not math.isnan(mtd) and mtd > 0 else math.nan)
    if math.isnan(denominator) or mdr >= 1.0:
        mtr = math.nan
    else:
        mtr = mtfa / (denominator * (1 - mdr)) if not math.isnan(mtfa) else math.nan
    return {"MDR": mdr, "MTD": mtd, "MTFA": mtfa, "MTR": mtr}


def compute_lpd(acc_d: float, acc0: float, n_drifts: int) -> float:
    """根据 Lukats 等的定义计算 lift-per-drift。"""
    if n_drifts <= 0:
        return 0.0
    return (acc_d - acc0) / float(n_drifts)

