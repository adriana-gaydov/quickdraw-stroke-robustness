"""Data download, validation, de-duplication, splitting, and loading (proposal, Section 3).

Only ``key_id``, ``word`` and ``drawing`` are kept in processed files; country codes,
timestamps and recognition flags are dropped and never used as features or filters.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .transforms import strokes_to_points


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def raw_path(category: str, raw_dir: Path = config.RAW_DIR) -> Path:
    return raw_dir / f"{category}.ndjson"


def download(categories=config.CATEGORIES, raw_dir: Path = config.RAW_DIR, force=False):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for c in categories:
        dest = raw_path(c, raw_dir)
        if dest.exists() and not force:
            print(f"[skip] {dest} exists")
            continue
        url = config.SOURCE_URL.format(category=c)
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Validation and identifiers
# ---------------------------------------------------------------------------

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def is_valid_drawing(drawing) -> bool:
    """Nonempty strokes, matching x/y lengths, integer coordinates in [0, 255]."""
    if not isinstance(drawing, list) or not drawing:
        return False
    for stroke in drawing:
        if not isinstance(stroke, list) or len(stroke) < 2:
            return False
        xs, ys = stroke[0], stroke[1]
        if not isinstance(xs, list) or not isinstance(ys, list):
            return False
        if len(xs) == 0 or len(xs) != len(ys):
            return False
        for v in (*xs, *ys):
            if not _is_int(v) or v < 0 or v > config.COORD_MAX:
                return False
    return True


def num_points(drawing) -> int:
    return sum(len(s[0]) for s in drawing)


def drawing_hash(drawing) -> bytes:
    """Hash of the full stroke structure (coordinates *and* stroke boundaries).

    Boundaries are part of the identity: the same points split differently draw different
    images (no line is drawn across a pen lift), so they are not duplicates.
    """
    return hashlib.sha1(json.dumps(drawing, separators=(",", ":")).encode()).digest()


def split_of(key_id: str) -> str:
    """Fixed identifier-based split: MD5(key_id) mod 10 -> 80/10/10.

    Raw key_ids are not uniform mod 10 (they appear to be even), so they are hashed first.
    """
    bucket = int(hashlib.md5(str(key_id).encode()).hexdigest(), 16) % 10
    for name, buckets in config.SPLIT_BUCKETS.items():
        if bucket in buckets:
            return name
    raise AssertionError(bucket)


def iter_raw(category: str, raw_dir: Path = config.RAW_DIR):
    """Yield (record or None, line_no) in source-file order; None marks unparsable lines."""
    with open(raw_path(category, raw_dir), encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                yield json.loads(line), i
            except json.JSONDecodeError:
                yield None, i


def _check(rec):
    """Return (status, drawing).

    status in {'ok', 'empty_or_malformed', 'bad_coordinates', 'too_long'}.
    """
    if not isinstance(rec, dict) or "key_id" not in rec:
        return "empty_or_malformed", None
    try:
        int(rec["key_id"])
    except (TypeError, ValueError):
        return "empty_or_malformed", None
    drawing = rec.get("drawing")
    if not isinstance(drawing, list) or not drawing:
        return "empty_or_malformed", None
    for stroke in drawing:
        if (not isinstance(stroke, list) or len(stroke) != 2
                or not isinstance(stroke[0], list) or not isinstance(stroke[1], list)
                or len(stroke[0]) == 0 or len(stroke[0]) != len(stroke[1])):
            return "empty_or_malformed", None
    if not is_valid_drawing(drawing):
        return "bad_coordinates", None
    if num_points(drawing) > config.MAX_POINTS:
        return "too_long", None
    return "ok", drawing


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------

def build_dataset(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR,
                  splits_dir: Path = config.SPLITS_DIR, per_category=config.PER_CATEGORY):
    """Three streaming passes over the raw files (keeps memory low).

    1. Group valid drawings by full stroke structure across all categories.
    2. Drop groups with conflicting labels; within same-label groups keep only the copy with
       the smallest key_id; assign each survivor a split from MD5(key_id).
    3. Per category and split, sort survivors by key_id, keep the first N, and collect them.
    """
    # Pass 1: duplicate groups -------------------------------------------------
    groups: dict[bytes, dict[str, int]] = defaultdict(dict)  # hash -> {label: min key}
    for c in config.CATEGORIES:
        for rec, _ in iter_raw(c, raw_dir):
            status, drawing = _check(rec)
            if status != "ok":
                continue
            h = drawing_hash(drawing)
            k = int(rec["key_id"])
            prev = groups[h].get(c)
            groups[h][c] = k if prev is None else min(prev, k)

    # Pass 2: exclusions and candidate pools ------------------------------------
    counts: dict[str, Counter] = {}
    chosen: dict[str, set[int]] = {}
    for c in config.CATEGORIES:
        cnt = Counter()
        pools = defaultdict(list)
        for rec, _ in iter_raw(c, raw_dir):
            cnt["loaded"] += 1
            status, drawing = _check(rec)
            if status != "ok":
                cnt[status] += 1
                continue
            g = groups[drawing_hash(drawing)]
            k = int(rec["key_id"])
            if len(g) > 1:
                cnt["conflicting_label_duplicate"] += 1
            elif k != g[c]:
                cnt["same_label_duplicate"] += 1
            else:
                pools[split_of(k)].append(k)
        chosen[c] = set()
        for s, n in per_category.items():
            pool = sorted(pools[s])
            if len(pool) < n:
                raise RuntimeError(f"{c}/{s}: only {len(pool)} of {n} drawings available")
            chosen[c].update(pool[:n])
        counts[c] = cnt
        print(f"{c}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))

    # Pass 3: collect the selected drawings, sorted by key_id -------------------
    selected = {s: [] for s in per_category}
    for c in config.CATEGORIES:
        recs = []
        for rec, _ in iter_raw(c, raw_dir):
            status, drawing = _check(rec)
            if status == "ok" and int(rec["key_id"]) in chosen[c]:
                recs.append((int(rec["key_id"]), drawing))
        for k, drawing in sorted(recs):
            selected[split_of(k)].append({"key_id": str(k), "word": c, "drawing": drawing})

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s, recs in selected.items():
        with open(out_dir / f"{s}.ndjson", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with open(splits_dir / f"{s}_ids.csv", "w", encoding="utf-8") as f:
            f.write("key_id,word\n")
            for r in recs:
                f.write(f"{r['key_id']},{r['word']}\n")
    with open(splits_dir / "filter_counts.json", "w", encoding="utf-8") as f:
        json.dump({c: dict(v) for c, v in counts.items()}, f, indent=2)
    return selected, counts


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Drawing:
    key_id: str
    label: int
    points: np.ndarray   # (T, 2) int64
    ends: np.ndarray     # (T,) bool


def load_split(split: str, processed_dir: Path = config.PROCESSED_DIR) -> list[Drawing]:
    out = []
    with open(processed_dir / f"{split}.ndjson", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pts, ends = strokes_to_points(r["drawing"])
            out.append(Drawing(r["key_id"], config.CATEGORY_TO_INDEX[r["word"]], pts, ends))
    return out
