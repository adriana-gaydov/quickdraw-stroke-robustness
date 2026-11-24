"""Sanity checks on the processed splits, plus the exclusion table for the report.

Run after ``scripts/prepare_data.py``:
    python scripts/check_data.py
Checks (stops with an error message if any fails):
  1. Each split has the expected number of drawings per category (4000 / 500 / 500).
  2. No key_id appears twice, within or across splits.
  3. Every drawing sits in the split that MD5(key_id) assigns it to.
  4. The id lists in splits/*.csv match the processed files exactly.
  5. Every drawing is valid and has at most MAX_POINTS points.
  6. No two kept drawings are identical (de-duplication held).
  7. Encoding (Eq. 1) round-trips: summing the scaled displacements gives back
     the original points, and the number of end flags equals the number of strokes.
Writes results/exclusions.md (exclusion counts per category and reason).
"""

import csv
import json
from collections import Counter

import _path  # noqa: F401
import numpy as np
from qdrobust import config
from qdrobust.data import drawing_hash, is_valid_drawing, num_points, split_of
from qdrobust.transforms import encode, strokes_to_points

REASONS = ["empty_or_malformed", "bad_coordinates", "too_long",
           "same_label_duplicate", "conflicting_label_duplicate"]


def load_processed(split):
    with open(config.PROCESSED_DIR / f"{split}.ndjson", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_ids(split):
    with open(config.SPLITS_DIR / f"{split}_ids.csv", encoding="utf-8") as f:
        return [(r["key_id"], r["word"]) for r in csv.DictReader(f)]


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def main():
    seen_ids = {}
    seen_hashes = set()

    for split, per_cat in config.PER_CATEGORY.items():
        recs = load_processed(split)

        # 1. counts per category
        counts = Counter(r["word"] for r in recs)
        for c in config.CATEGORIES:
            check(counts[c] == per_cat,
                  f"{split}/{c} has {counts[c]} drawings, expected {per_cat}")
        check(set(counts) == set(config.CATEGORIES), f"{split} has unknown categories")

        for r in recs:
            k = r["key_id"]
            # 2. unique ids
            check(k not in seen_ids, f"key_id {k} appears in {seen_ids.get(k)} and {split}")
            seen_ids[k] = split
            # 3. correct split
            check(split_of(k) == split, f"key_id {k} is in {split} but hashes to {split_of(k)}")
            # 5. valid and short enough
            d = r["drawing"]
            check(is_valid_drawing(d), f"key_id {k} is not a valid drawing")
            check(num_points(d) <= config.MAX_POINTS, f"key_id {k} is longer than {config.MAX_POINTS}")
            # 6. no duplicates kept
            h = drawing_hash(d)
            check(h not in seen_hashes, f"key_id {k} duplicates another kept drawing")
            seen_hashes.add(h)
            # 7. encoding round trip
            points, ends = strokes_to_points(d)
            u = encode(points, ends)
            rebuilt = np.rint(np.cumsum(u[:, :2] * config.COORD_MAX, axis=0)).astype(np.int64)
            check(np.array_equal(rebuilt, points), f"key_id {k}: encoding does not round-trip")
            check(int(u[:, 2].sum()) == len(d), f"key_id {k}: end flags != number of strokes")
            check(u[-1, 2] == 1.0, f"key_id {k}: final point is not marked as a stroke end")

        # 4. id lists match the processed file
        check(load_ids(split) == [(r["key_id"], r["word"]) for r in recs],
              f"splits/{split}_ids.csv does not match data/processed/{split}.ndjson")
        print(f"{split}: {len(recs)} drawings OK")

    # Exclusion table ------------------------------------------------------------
    with open(config.SPLITS_DIR / "filter_counts.json", encoding="utf-8") as f:
        fc = json.load(f)
    header = "| Category | Loaded | " + " | ".join(REASONS) + " | Total excluded |"
    lines = [header, "|" + "---|" * (len(REASONS) + 3)]
    totals = Counter()
    for c in config.CATEGORIES:
        row = [fc[c].get(reason, 0) for reason in REASONS]
        totals.update({"loaded": fc[c]["loaded"], **dict(zip(REASONS, row))})
        lines.append(f"| {c} | {fc[c]['loaded']:,} | " + " | ".join(str(v) for v in row)
                     + f" | {sum(row)} |")
    lines.append(f"| **All** | {totals['loaded']:,} | "
                 + " | ".join(str(totals[r]) for r in REASONS)
                 + f" | {sum(totals[r] for r in REASONS)} |")
    table = "\n".join(lines)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "exclusions.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print("\nAll checks passed. Exclusion table saved to results/exclusions.md")


if __name__ == "__main__":
    main()