"""
Matplotlib visualizations with dark theme.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Theme constants
BG_COLOR = "#0f0f0f"
ACCENT = "#e85d04"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a2a"
SECONDARY_COLORS = ["#e85d04", "#ff9e00", "#dc2f02", "#9d0208", "#f48c06", "#ffba08"]


def _require_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization. Install with: pip install apple-health-explorer[viz]")


def _apply_dark_theme(ax):
    """Apply dark theme to an axes object."""
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)


class HealthVisualizer:
    """Visualization layer for parsed health data."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    def plot_steps(self, ax=None):
        """Plot daily steps with 7-day rolling average."""
        _require_mpl()

        df = self.data.get("steps")
        if df is None or df.empty:
            return None

        daily = df.groupby(df.index.date)["value"].sum()
        daily.index = pd.to_datetime(daily.index)
        rolling = daily.rolling(7, min_periods=1).mean()

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))

        ax.bar(daily.index, daily.values, color=ACCENT, alpha=0.3, width=1)
        ax.plot(rolling.index, rolling.values, color=ACCENT, linewidth=2, label="Media movil 7d")
        ax.set_ylabel("Pasos")
        ax.set_title("Pasos diarios")
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        _apply_dark_theme(ax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.tight_layout()
        return ax

    def plot_sleep(self, ax=None):
        """Plot nightly sleep duration."""
        _require_mpl()

        df = self.data.get("sleep")
        if df is None or df.empty:
            return None

        sleep_vals = {
            "HKCategoryValueSleepAnalysisAsleepUnspecified",
            "HKCategoryValueSleepAnalysisAsleepCore",
            "HKCategoryValueSleepAnalysisAsleepDeep",
            "HKCategoryValueSleepAnalysisAsleepREM",
        }
        s = df[df["value"].isin(sleep_vals)].copy()
        s["hours"] = (s["end"] - s.index).dt.total_seconds() / 3600
        s["night"] = s.index.normalize()
        mask = s.index.hour < 12
        s.loc[mask, "night"] = s.loc[mask, "night"] - pd.Timedelta(days=1)
        nightly = s.groupby("night")["hours"].sum()

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))

        rolling = nightly.rolling(7, min_periods=1).mean()
        ax.fill_between(nightly.index, nightly.values, color=ACCENT, alpha=0.2)
        ax.plot(rolling.index, rolling.values, color=ACCENT, linewidth=2, label="Media movil 7d")
        ax.axhline(y=8, color=TEXT_COLOR, linestyle="--", alpha=0.3, label="8h objetivo")
        ax.set_ylabel("Horas de sueno")
        ax.set_title("Sueno por noche")
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
        _apply_dark_theme(ax)
        plt.tight_layout()
        return ax

    def plot_heart(self, ax=None):
        """Plot resting HR and HRV trends."""
        _require_mpl()

        rhr = self.data.get("resting_hr")
        hrv = self.data.get("hrv")
        has_rhr = rhr is not None and not rhr.empty
        has_hrv = hrv is not None and not hrv.empty

        if not has_rhr and not has_hrv:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))

        if has_rhr:
            daily_rhr = rhr.groupby(rhr.index.date)["value"].mean()
            daily_rhr.index = pd.to_datetime(daily_rhr.index)
            rolling_rhr = daily_rhr.rolling(7, min_periods=1).mean()
            ax.plot(rolling_rhr.index, rolling_rhr.values, color=ACCENT, linewidth=2, label="FC reposo (bpm)")
            ax.set_ylabel("FC reposo (bpm)")

        if has_hrv:
            daily_hrv = hrv.groupby(hrv.index.date)["value"].mean()
            daily_hrv.index = pd.to_datetime(daily_hrv.index)
            rolling_hrv = daily_hrv.rolling(7, min_periods=1).mean()
            ax2 = ax.twinx()
            ax2.plot(rolling_hrv.index, rolling_hrv.values, color="#ff9e00", linewidth=2, label="HRV (ms)")
            ax2.set_ylabel("HRV (ms)")
            ax2.tick_params(colors=TEXT_COLOR)
            ax2.yaxis.label.set_color(TEXT_COLOR)
            for spine in ax2.spines.values():
                spine.set_color(GRID_COLOR)

        ax.set_title("Frecuencia cardiaca y HRV (media movil 7d)")
        _apply_dark_theme(ax)
        plt.tight_layout()
        return ax

    def plot_correlation_matrix(self, ax=None):
        """Plot heatmap of cross-metric correlations."""
        _require_mpl()

        from .analyzer import HealthAnalyzer

        corr = HealthAnalyzer(self.data).correlation_matrix()
        if corr.empty:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(corr.values, cmap="YlOrRd", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)

        # Annotate cells
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                color = "#000000" if abs(val) < 0.5 else "#ffffff"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_title("Correlaciones entre metricas")
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        _apply_dark_theme(ax)
        plt.tight_layout()
        return ax
