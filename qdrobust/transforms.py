"""Sequence representation and controlled stroke-boundary transformations.

A drawing is stored as a pair ``(points, ends)``:

* ``points``: int array of shape (T, 2) with absolute (x, y) coordinates in [0, 255].
* ``ends``:   bool array of shape (T,), True where a stroke ends (including the final point).

Transformations (proposal, Section 4.1):

* ``fragment(points, ends, J)``  -- F_J: split strokes at interior points J, duplicating
  the shared endpoint. The first copy ends a stroke, the second does not.
* ``duplicate(points, ends, J)`` -- D_J: insert the same duplicate points without new
  boundaries. D_J(U) and F_J(U) have identical coordinates and lengths.
* ``canonicalize(points, ends)`` -- C: join consecutive strokes whose endpoints are exactly
  shared (keeping one copy) and collapse consecutive duplicate coordinates within strokes.
"""

from __future__ import annotations

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def strokes_to_points(strokes) -> tuple[np.ndarray, np.ndarray]:
    """QuickDraw ``[[xs, ys], ...]`` strokes -> ``(points, ends)``."""
    pts, ends = [], []
    for xs, ys in strokes:
        s = np.stack([np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)], axis=1)
        e = np.zeros(len(s), dtype=bool)
        e[-1] = True
        pts.append(s)
        ends.append(e)
    return np.concatenate(pts), np.concatenate(ends)


def points_to_strokes(points: np.ndarray, ends: np.ndarray) -> list[np.ndarray]:
    """``(points, ends)`` -> list of (n_i, 2) arrays, one per stroke."""
    cut = np.flatnonzero(ends) + 1
    return [s for s in np.split(points, cut) if len(s)]


def encode(points: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Eq. (1): u_t = [dx/255, dy/255, b_t] with p_0 = (0, 0). Returns float32 (T, 3)."""
    prev = np.vstack([np.zeros((1, 2), dtype=points.dtype), points[:-1]])
    delta = (points - prev).astype(np.float32) / config.COORD_MAX
    return np.concatenate([delta, ends.astype(np.float32)[:, None]], axis=1)


def num_strokes(ends: np.ndarray) -> int:
    return int(ends.sum())


# ---------------------------------------------------------------------------
# Eligible split locations and masks
# ---------------------------------------------------------------------------

def eligible_points(ends: np.ndarray) -> np.ndarray:
    """Indices of interior stroke points (neither the first nor the last point of a stroke).

    Splitting at such a point keeps both adjacent segments, so no line segment is removed.
    """
    starts = np.zeros_like(ends)
    starts[0] = True
    starts[1:] = ends[:-1]
    return np.flatnonzero(~starts & ~ends)


def select_points(eligible: np.ndarray, uniforms: np.ndarray, q: float) -> np.ndarray:
    """Select eligible points whose uniform value is below q (nested in q for fixed uniforms)."""
    if q >= 1.0:
        return eligible.copy()
    return eligible[uniforms < q]


def item_rng(*keys: int) -> np.random.Generator:
    """Deterministic generator keyed by integers (e.g. seed, epoch, item index)."""
    return np.random.default_rng([int(k) & 0xFFFFFFFF for k in keys])


# ---------------------------------------------------------------------------
# F_J, D_J, C
# ---------------------------------------------------------------------------

def _insert_duplicates(points: np.ndarray, ends: np.ndarray, J: np.ndarray):
    J = np.unique(np.asarray(J, dtype=np.int64))
    repeats = np.ones(len(points), dtype=np.int64)
    repeats[J] = 2
    new_points = np.repeat(points, repeats, axis=0)
    new_ends = np.repeat(ends, repeats)
    first_copy = J + np.arange(len(J))  # position of the first copy of each p_j
    return new_points, new_ends, first_copy


def fragment(points: np.ndarray, ends: np.ndarray, J) -> tuple[np.ndarray, np.ndarray]:
    """F_J: (..., p_j, p_{j+1}, ...) -> (..., p_j) | (p_j, p_{j+1}, ...)."""
    new_points, new_ends, first_copy = _insert_duplicates(points, ends, J)
    new_ends[first_copy] = True
    return new_points, new_ends


def duplicate(points: np.ndarray, ends: np.ndarray, J) -> tuple[np.ndarray, np.ndarray]:
    """D_J: duplicate p_j for j in J without adding a boundary."""
    new_points, new_ends, _ = _insert_duplicates(points, ends, J)
    return new_points, new_ends


def canonicalize(points: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C: join strokes with exactly shared endpoints, then collapse within-stroke duplicates."""
    joined: list[np.ndarray] = []
    for s in points_to_strokes(points, ends):
        if joined and np.array_equal(joined[-1][-1], s[0]):
            joined[-1] = np.concatenate([joined[-1], s[1:]])
        else:
            joined.append(s)
    out_pts, out_ends = [], []
    for s in joined:
        keep = np.ones(len(s), dtype=bool)
        keep[1:] = np.any(s[1:] != s[:-1], axis=1)
        s = s[keep]
        e = np.zeros(len(s), dtype=bool)
        e[-1] = True
        out_pts.append(s)
        out_ends.append(e)
    return np.concatenate(out_pts), np.concatenate(out_ends)


# ---------------------------------------------------------------------------
# Invariance checks
# ---------------------------------------------------------------------------

def drawn_segments(points: np.ndarray, ends: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Ordered list of nonzero-length segments drawn within strokes."""
    segs = []
    for s in points_to_strokes(points, ends):
        for a, b in zip(s[:-1], s[1:]):
            if not np.array_equal(a, b):
                segs.append((int(a[0]), int(a[1]), int(b[0]), int(b[1])))
    return segs


def same_sequence(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
