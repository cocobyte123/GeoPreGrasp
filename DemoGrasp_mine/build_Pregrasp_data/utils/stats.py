"""Statistics helpers for rollout-to-field aggregation."""

from __future__ import annotations

import numpy as np


def aggregate_counts(mtp_indices: np.ndarray, success: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    mtp = np.asarray(mtp_indices, dtype=np.int64).reshape(-1, 3)
    success = np.asarray(success).reshape(-1).astype(np.float32)
    attempt_count = np.zeros(shape, dtype=np.float32)
    success_count = np.zeros(shape, dtype=np.float32)
    for (m_id, t_id, p_id), value in zip(mtp, success):
        if 0 <= m_id < shape[0] and 0 <= t_id < shape[1] and 0 <= p_id < shape[2]:
            attempt_count[m_id, t_id, p_id] += 1.0
            success_count[m_id, t_id, p_id] += float(value)
    return attempt_count, success_count


def target_and_confidence(attempt_count: np.ndarray, success_count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    attempt_count = np.asarray(attempt_count, dtype=np.float32)
    success_count = np.asarray(success_count, dtype=np.float32)
    target = np.zeros_like(attempt_count, dtype=np.float32)
    nonzero = attempt_count > 0.0
    target[nonzero] = success_count[nonzero] / attempt_count[nonzero]
    max_attempt = float(attempt_count.max())
    confidence = np.zeros_like(attempt_count, dtype=np.float32)
    if max_attempt > 0.0:
        confidence = attempt_count / max_attempt
    return target.astype(np.float32), confidence.astype(np.float32)

