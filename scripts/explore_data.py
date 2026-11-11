"""Exploratory statistics (proposal Table 1 / Figure 1) on training-assigned drawings.

Usage:
    python scripts/explore_data.py --per-category 500     # proposal sample
    python scripts/explore_data.py --per-category 4000    # full training split
Writes results/eda_<n>.json and figures/eda_<n>.png.
"""

import argparse
import json

import _path  # noqa: F401
import numpy as np
from qdrobust import config
from qdrobust.data import load_split
from qdrobust.transforms import eligible_points, num_strokes


def summarize(values):
    v = np.asarray(values)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "min": int(v.min()), "max": int(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-category", type=int, default=500)
    args = ap.parse_args()

    train = load_split("train")
    taken = {c: 0 for c in range(config.NUM_CLASSES)}
    sample = []
    for d in train:  # processed file keeps source-file order within each category
        if taken[d.label] < args.per_category:
            taken[d.label] += 1
            sample.append(d)

    pts = [len(d.points) for d in sample]
    strokes = [num_strokes(d.ends) for d in sample]
    elig = [len(eligible_points(d.ends)) for d in sample]
    stats = {
        "n": len(sample),
        "points": summarize(pts),
        "strokes": summarize(strokes),
        "eligible_split_locations": summarize(elig),
        "no_eligible_point": int(sum(e == 0 for e in elig)),
        "median_strokes_by_category": {
            c: float(np.median([s for s, d in zip(strokes, sample) if d.label == i]))
            for i, c in enumerate(config.CATEGORIES)
        },
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"eda_{args.per_category}.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.hist(pts, bins=np.arange(0, config.MAX_POINTS + 25, 25), color="#2f5f8a")
    a.set(title="Sequence length", xlabel="Points per drawing", ylabel="Drawings")
    b.boxplot([[s for s, d in zip(strokes, sample) if d.label == i]
               for i in range(config.NUM_CLASSES)], showfliers=False)
    b.set_xticks(range(1, config.NUM_CLASSES + 1), config.CATEGORIES, rotation=45)
    b.set(title="Stroke-count variation", ylabel="Strokes per drawing")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"eda_{args.per_category}.png", dpi=150)


if __name__ == "__main__":
    main()
