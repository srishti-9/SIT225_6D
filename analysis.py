"""analysis of labelled 10-second accelerometer windows.

Inputs (in the working directory unless overridden on the command line):
    activity_data/<seq>_<yyyymmddHHMMSS>.csv   accelerometer samples for one window
    activity_data/<seq>_<yyyymmddHHMMSS>.jpg   webcam image for the same window
    labels.json                                draft label and flags for each window
    phone_visible.json                         windows where the phone is visible

Outputs:
    annotation.csv, annotation_notes.csv, spot_check.csv, class_features.csv,
    results.json and figures/*.png

"Movement" is range(x) + range(y) + range(z) within a window, in g. It is 0 for a
perfectly still phone.

Usage:
    python analysis.py [--data activity_data] [--out .]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu

# settings
SESSION_GAP_S = 15  # a gap longer than this between saved windows starts a new session
STILL_G = 0.05  # movement below this counts as a still phone
LARGE_G = 0.5  # movement above this counts as large movement
SPOT_CHECK_N = 60  # windows in the random sample for the manual label check
SPOT_CHECK_SEED = 225

CLASS_NAMES = {0: "0 no activity", 1: "1 hair touching", 2: "2 hand to face"}
CLASS_COLOURS = {0: "#6b7280", 1: "#d9480f", 2: "#1c7ed6"}
AXIS_COLOURS = {"x": "#e03131", "y": "#2f9e44", "z": "#1971c2"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


#  data loading
def read_window(data_dir: Path, name: str) -> pd.DataFrame:
    """Return the samples (timestamp, x, y, z) of one window."""
    return pd.read_csv(data_dir / f"{name}.csv")


def load_windows(data_dir: Path) -> pd.DataFrame:
    """Compute per-window features for every <seq>_<timestamp>.csv file."""
    files = sorted(data_dir.glob("*.csv"), key=lambda p: int(p.stem.split("_")[0]))
    rows = []
    for path in files:
        seq, saved = path.stem.split("_")
        d = pd.read_csv(path)
        magnitude = np.sqrt(d.x**2 + d.y**2 + d.z**2)
        rows.append(
            {
                "name": path.stem,
                "seq": int(seq),
                "saved": pd.to_datetime(saved, format="%Y%m%d%H%M%S"),
                "n": len(d),
                "mx": d.x.mean(),
                "my": d.y.mean(),
                "mz": d.z.mean(),
                "sx": d.x.std(ddof=0),
                "sy": d.y.std(ddof=0),
                "sz": d.z.std(ddof=0),
                "rx": np.ptp(d.x),
                "ry": np.ptp(d.y),
                "rz": np.ptp(d.z),
                "mag": magnitude.mean(),
                "magrange": np.ptp(magnitude),
            }
        )
    windows = pd.DataFrame(rows)
    windows["move"] = windows.rx + windows.ry + windows.rz
    gap = windows.saved.diff().dt.total_seconds()
    windows["session"] = (gap > SESSION_GAP_S).cumsum() + 1
    return windows


def attach_labels(windows: pd.DataFrame, labels: dict, phone_visible: list) -> pd.DataFrame:
    """Add label and flag columns.

    Flags: h = head/body movement, u = uncertain, o = person out of frame.
    """
    windows = windows.copy()
    for key in ("label", "h", "u", "o"):
        windows[key] = [labels[str(seq)][key] for seq in windows.seq]
    windows["label"] = windows.label.astype(int)
    windows["phone_visible"] = windows.seq.isin(phone_visible)
    return windows


#  output files
def write_annotation_files(windows: pd.DataFrame, out_dir: Path) -> None:
    """Write the two-column annotation file, the notes file and the spot-check sheet."""
    windows[["name", "label"]].rename(columns={"name": "filename"}).to_csv(
        out_dir / "annotation.csv", index=False
    )

    notes = windows.assign(
        head_move=windows.h.astype(int),
        uncertain=windows.u.astype(int),
        out_of_frame=windows.o.astype(int),
        phone_visible=windows.phone_visible.astype(int),
    )[
        [
            "name",
            "label",
            "head_move",
            "uncertain",
            "out_of_frame",
            "phone_visible",
            "n",
            "move",
            "mag",
        ]
    ]
    notes = notes.rename(
        columns={"name": "filename", "move": "movement_g", "mag": "mean_magnitude_g"}
    )
    notes.round(4).to_csv(out_dir / "annotation_notes.csv", index=False)

    sample = windows.sample(SPOT_CHECK_N, random_state=SPOT_CHECK_SEED).sort_values("seq")
    pd.DataFrame(
        {"filename": sample.name, "draft_label": sample.label, "my_label": "", "agree": ""}
    ).to_csv(out_dir / "spot_check.csv", index=False)


#  class summary
def median_iqr(values: pd.Series) -> str:
    """Format as 'median [Q1-Q3]'."""
    q1, med, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{med:.3f} [{q1:.3f}-{q3:.3f}]"


def class_feature_table(windows: pd.DataFrame, in_view_no_activity: pd.DataFrame) -> list[dict]:
    """One row of summary features for each class (label 0 is shown twice)."""
    groups = {
        "0 (all)": windows[windows.label == 0],
        "0 (person in view)": in_view_no_activity,
        "1": windows[windows.label == 1],
        "2": windows[windows.label == 2],
    }
    return [
        {
            "class": name,
            "windows": len(g),
            "std_x": round(g.sx.median(), 4),
            "std_y": round(g.sy.median(), 4),
            "std_z": round(g.sz.median(), 4),
            "mean_magnitude": median_iqr(g.mag),
            "magnitude_range": median_iqr(g.magrange),
            "movement": median_iqr(g.move),
            "pct_move_gt_0.05": round(100 * (g.move > STILL_G).mean(), 1),
            "pct_move_gt_0.5": round(100 * (g.move > LARGE_G).mean(), 1),
        }
        for name, g in groups.items()
    ]


def mann_whitney(a: pd.Series, b: pd.Series) -> dict:
    """Probability that a random value of `a` exceeds one of `b` (0.5 = no separation)."""
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"auc": u / (len(a) * len(b)), "p": p}


def best_threshold(in_view: pd.DataFrame) -> dict:
    """Best single movement threshold for 'no activity' vs 'any activity'."""
    has_activity = in_view.label > 0
    scores = [(t, ((in_view.move > t) == has_activity).mean()) for t in np.linspace(0.005, 3, 600)]
    threshold, accuracy = max(scores, key=lambda s: s[1])
    majority = max((~has_activity).mean(), has_activity.mean())
    return {
        "thr": round(threshold, 3),
        "acc": round(accuracy, 3),
        "base": round(majority, 3),
        "n": len(in_view),
    }


def session_table(windows: pd.DataFrame) -> pd.DataFrame:
    """Per-session window counts, label counts, out-of-frame counts and still share."""
    return windows.groupby("session").agg(
        first=("seq", "min"),
        last=("seq", "max"),
        start=("saved", "min"),
        end=("saved", "max"),
        n=("seq", "count"),
        l0=("label", lambda s: (s == 0).sum()),
        l1=("label", lambda s: (s == 1).sum()),
        l2=("label", lambda s: (s == 2).sum()),
        o=("o", "sum"),
        still=("move", lambda s: (s < STILL_G).mean()),
    )


def compute_results(windows: pd.DataFrame, sessions: pd.DataFrame) -> dict:
    """Collect every number quoted in the report into one dictionary."""
    w = windows
    c1, c2 = w[w.label == 1], w[w.label == 2]
    in_view = w[~w.o]
    vis0 = w[(w.label == 0) & ~w.o]  # no activity, person in view
    pairs = {"0v1": (vis0, c1), "0v2": (vis0, c2), "1v2": (c1, c2)}
    activity = w.label > 0
    large = w[w.move > LARGE_G]

    return {
        "n": len(w),
        "counts": w.label.value_counts().sort_index().to_dict(),
        "table": class_feature_table(w, vis0),
        "auc": {k: mann_whitney(a.move, b.move) for k, (a, b) in pairs.items()},
        # same comparison without the windows flagged as uncertain
        "auc_clear": {k: mann_whitney(a[~a.u].move, b[~b.u].move) for k, (a, b) in pairs.items()},
        "n_clear": {"0": [len(g[~g.u]) for g in (vis0, c1, c2)]},
        "n_flags": {
            "o": int(w.o.sum()),
            "u": int(w.u.sum()),
            "h": int(w.h.sum()),
            "phone": int(w.phone_visible.sum()),
        },
        "u_by_label": w[w.u].label.value_counts().sort_index().to_dict(),
        "o_labels": w[w.o].label.value_counts().to_dict(),
        "act_still": {"n": int((activity & (w.move < STILL_G)).sum()), "of": int(activity.sum())},
        "act_still02": int((activity & (w.move < 0.02)).sum()),
        "noact_moved": {"n": int((vis0.move > LARGE_G).sum()), "of": len(vis0)},
        "noact_moved_seq": vis0[vis0.move > LARGE_G]
        .sort_values("move", ascending=False)
        .seq.head(12)
        .tolist(),
        "big_windows": {
            "n": len(large),
            "by_label": large.label.value_counts().sort_index().to_dict(),
            "n_h": int(large.h.sum()),
            "n_o": int(large.o.sum()),
        },
        "still_windows": {
            "n": int((w.move < STILL_G).sum()),
            "by_label": w[w.move < STILL_G].label.value_counts().sort_index().to_dict(),
        },
        "best_thr": best_threshold(in_view),
        "sessions": sessions.reset_index().astype({"start": str, "end": str}).to_dict("records"),
        "u_move": {
            "n": int(w.u.sum()),
            "gt005": int(((w.move > STILL_G) & w.u).sum()),
            "gt05": int(((w.move > LARGE_G) & w.u).sum()),
        },
        "window_sample_stats": {
            "mean_n": round(w.n.mean(), 2),
            "counts": w.n.value_counts().sort_index().to_dict(),
        },
        "mag_share": round(((w.mag > 0.95) & (w.mag < 1.05)).mean(), 3),
    }


# ----------------------------------------------------------------------------- figures
def elapsed_seconds(d: pd.DataFrame) -> np.ndarray:
    """Seconds since the first sample. Falls back to sample index if times can't be parsed."""
    first = str(d.timestamp.iloc[0])
    time_only = "." in first and len(first) < 20  # early windows store the time of day only
    parsed = pd.to_datetime(
        d.timestamp, format="%H:%M:%S.%f" if time_only else None, errors="coerce"
    )
    if parsed.notna().all():
        return (parsed - parsed.iloc[0]).dt.total_seconds().values
    return np.arange(len(d), dtype=float)


def deviation_limit(data_dir: Path, rows: list) -> float:
    """Common y-limit (g) so windows in one figure share the same scale."""
    largest = 0.0
    for row in rows:
        d = read_window(data_dir, row["name"])[list("xyz")]
        largest = max(largest, np.abs(d.sub(d.mean()).values).max())
    return max(0.02, largest * 1.25)


def plot_window(ax_image, ax_graph, data_dir: Path, row, limit: float) -> None:
    """Draw one window: webcam image on top, x/y/z deviation from the window mean below."""
    ax_image.imshow(Image.open(data_dir / f"{row['name']}.jpg"))
    ax_image.axis("off")
    ax_image.set_title(
        f"{row['name']}\nlabel {int(row['label'])}, movement {row['move']:.3f} g", fontsize=7.5
    )
    d = read_window(data_dir, row["name"])
    t = elapsed_seconds(d)
    for axis, colour in AXIS_COLOURS.items():
        ax_graph.plot(t, d[axis] - d[axis].mean(), "o-", color=colour, label=axis, ms=3.5, lw=1.2)
    ax_graph.set_xlim(-0.5, 10.5)
    ax_graph.set_ylim(-limit, limit)
    ax_graph.set_xlabel("seconds into window")


def plot_window_row(data_dir: Path, rows: list, path: Path) -> None:
    """Save a figure with one column per window."""
    fig, ax = plt.subplots(
        2, len(rows), figsize=(3.6 * len(rows), 4.6), gridspec_kw={"height_ratios": [1.15, 1]}
    )
    limit = deviation_limit(data_dir, rows)
    for j, row in enumerate(rows):
        plot_window(ax[0, j], ax[1, j], data_dir, row, limit)
        if j == 0:
            ax[1, j].set_ylabel("deviation from window mean (g)")
        else:
            ax[1, j].set_yticklabels([])
    ax[1, 0].legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_windows(df: pd.DataFrame, quantiles=(0.25, 0.5, 0.9)) -> list:
    """Windows at fixed movement percentiles, ignoring uncertain/head-movement/out-of-frame."""
    clear = df[~df.u & ~df.h & ~df.o].sort_values("move")
    return [clear.iloc[round(q * (len(clear) - 1))] for q in quantiles]


def plot_session_overview(windows: pd.DataFrame, sessions: pd.DataFrame, path: Path) -> None:
    """Window means, movement coloured by label, and the label sequence."""
    fig, ax = plt.subplots(
        3, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 0.35]}
    )

    for axis, colour in AXIS_COLOURS.items():
        ax[0].plot(windows.seq, windows["m" + axis], color=colour, lw=0.8, label=axis)
    ax[0].set_ylabel("window mean (g)")
    ax[0].legend(ncol=3, frameon=False, fontsize=8, loc="lower right")

    ax[1].scatter(
        windows.seq,
        windows.move.clip(lower=1e-3),
        c=[CLASS_COLOURS[label] for label in windows.label],
        s=9,
    )
    ax[1].set_yscale("log")
    ax[1].set_ylabel("movement (g, log scale)")
    ax[1].axhline(STILL_G, color="k", lw=0.6, ls=":")
    ax[1].text(2, 0.06, f"{STILL_G} g", fontsize=7)
    for label, name in CLASS_NAMES.items():
        ax[1].scatter([], [], c=CLASS_COLOURS[label], s=12, label=name)
    ax[1].legend(frameon=False, fontsize=7, ncol=3, loc="upper right")

    for seq, label in zip(windows.seq, windows.label):
        ax[2].axvspan(seq - 0.5, seq + 0.5, color=CLASS_COLOURS[label], lw=0)
    ax[2].set_yticks([])
    ax[2].set_xlabel("window sequence number")
    ax[2].set_ylabel("label", rotation=0, labelpad=18, va="center")

    starts = sessions["first"].tolist()
    for a in ax[:2]:
        for start in starts[1:]:
            a.axvline(start - 0.5, color="k", ls=":", lw=0.7)
    for i, start in enumerate(starts):
        if i in (2, 3):  # sessions 3 and 4 are one window each; S2-S4 share one label
            continue
        text, align, offset = ("S2-S4", "right", -1) if i == 1 else (f"S{i + 1}", "left", 1)
        ax[0].text(start + offset, ax[0].get_ylim()[1] * 0.98, text, fontsize=7, va="top", ha=align)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_movement_by_class(groups: list, path: Path) -> None:
    """Box plot with the individual windows overlaid, log scale."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    data = [g.move.clip(lower=1e-3) for g in groups]
    box = ax.boxplot(
        data, widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": "k"}
    )
    for patch, label in zip(box["boxes"], (0, 1, 2)):
        patch.set_facecolor(CLASS_COLOURS[label])
        patch.set_alpha(0.35)
    for i, values in enumerate(data):
        jitter = np.random.RandomState(i).normal(i + 1, 0.06, len(values))
        ax.scatter(jitter, values, s=5, color=CLASS_COLOURS[i], alpha=0.5)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        [
            f"0 no activity\n(person in view, n={len(groups[0])})",
            f"1 hair touching\n(n={len(groups[1])})",
            f"2 hand to face\n(n={len(groups[2])})",
        ],
        fontsize=8,
    )
    ax.set_ylabel("movement (g, log scale)")
    ax.axhline(STILL_G, color="k", lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("activity_data"),
        help="folder with the <seq>_<timestamp>.csv/.jpg files",
    )
    parser.add_argument("--out", type=Path, default=Path("."), help="output folder")
    args = parser.parse_args()

    data_dir, out_dir = args.data, args.out
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads(Path("labels.json").read_text())
    phone_visible = json.loads(Path("phone_visible.json").read_text())["phone_visible"]
    windows = attach_labels(load_windows(data_dir), labels, phone_visible)
    sessions = session_table(windows)

    write_annotation_files(windows, out_dir)
    results = compute_results(windows, sessions)

    vis0 = windows[(windows.label == 0) & ~windows.o]
    class1, class2 = windows[windows.label == 1], windows[windows.label == 2]

    plot_window_row(
        data_dir, [windows.iloc[i] for i in range(3)], fig_dir / "fig2_first_three_windows.png"
    )
    picked = {k: pick_windows(windows[windows.label == k]) for k in (0, 1, 2)}
    results["picked"] = {
        k: [(int(r.seq), r["name"], round(float(r.move), 3), int(r.n)) for r in rows]
        for k, rows in picked.items()
    }
    for k, rows in picked.items():
        plot_window_row(data_dir, rows, fig_dir / f"fig_pattern_class{k}.png")
    plot_session_overview(windows, sessions, fig_dir / "fig3_session_overview.png")
    plot_movement_by_class([vis0, class1, class2], fig_dir / "fig4_movement_by_class.png")

    pd.DataFrame(results["table"]).to_csv(out_dir / "class_features.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print("ok")


if __name__ == "__main__":
    main()