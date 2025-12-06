"""
Analysis functions for Apple Health data.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class HealthAnalyzer:
    """Run analyses on parsed Apple Health DataFrames."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def steps_analysis(self) -> Dict[str, pd.DataFrame]:
        """Daily, weekly and monthly step patterns plus trend."""
        df = self.data.get("steps")
        if df is None or df.empty:
            return {"daily": pd.DataFrame(), "weekly": pd.DataFrame(), "monthly": pd.DataFrame()}

        daily = df.groupby(df.index.date)["value"].sum().rename_axis("date")
        daily.index = pd.to_datetime(daily.index)

        weekly = daily.resample("W").agg(["mean", "sum", "std"]).round(0)
        monthly = daily.resample("ME").agg(["mean", "sum", "std"]).round(0)

        # 7-day rolling trend
        trend = daily.rolling(7, min_periods=1).mean().rename("rolling_7d")

        return {
            "daily": daily.to_frame("steps"),
            "weekly": weekly,
            "monthly": monthly,
            "trend": trend.to_frame(),
        }

    # ------------------------------------------------------------------
    # Sleep
    # ------------------------------------------------------------------

    def sleep_analysis(self) -> Dict[str, pd.DataFrame]:
        """Sleep duration, consistency and quality metrics."""
        df = self.data.get("sleep")
        if df is None or df.empty:
            return {"nightly": pd.DataFrame(), "weekly": pd.DataFrame()}

        # Map Apple sleep values to categories
        sleep_map = {
            "HKCategoryValueSleepAnalysisInBed": "in_bed",
            "HKCategoryValueSleepAnalysisAsleepUnspecified": "asleep",
            "HKCategoryValueSleepAnalysisAsleepCore": "core",
            "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
            "HKCategoryValueSleepAnalysisAsleepREM": "rem",
            "HKCategoryValueSleepAnalysisAwake": "awake",
        }

        df = df.copy()
        df["category"] = df["value"].map(sleep_map).fillna("other")
        df["duration_hours"] = (df["end"] - df.index).dt.total_seconds() / 3600

        # Assign each record to its sleep night (date of the start, shifted if after midnight)
        df["night"] = df.index.normalize()
        mask_after_midnight = df.index.hour < 12
        df.loc[mask_after_midnight, "night"] = df.loc[mask_after_midnight, "night"] - pd.Timedelta(days=1)

        # Nightly aggregation
        nightly = df.groupby(["night", "category"])["duration_hours"].sum().unstack(fill_value=0)

        if "in_bed" not in nightly.columns:
            nightly["in_bed"] = 0
        sleep_cols = [c for c in ["core", "deep", "rem", "asleep"] if c in nightly.columns]
        nightly["total_sleep"] = nightly[sleep_cols].sum(axis=1) if sleep_cols else 0

        # Consistency: std of bedtime hour
        bedtimes = df.groupby("night").apply(
            lambda g: g.index.min().hour + g.index.min().minute / 60, include_groups=False
        )
        nightly["bedtime_hour"] = bedtimes

        weekly = nightly.resample("W").mean().round(2)

        return {"nightly": nightly, "weekly": weekly}

    # ------------------------------------------------------------------
    # Heart
    # ------------------------------------------------------------------

    def heart_analysis(self) -> Dict[str, pd.DataFrame]:
        """Resting heart rate trends and HRV patterns."""
        result = {}

        # Resting HR — typically one reading per day
        rhr = self.data.get("resting_hr")
        if rhr is not None and not rhr.empty:
            daily_rhr = rhr.groupby(rhr.index.date)["value"].mean().rename_axis("date")
            daily_rhr.index = pd.to_datetime(daily_rhr.index)
            result["resting_hr_daily"] = daily_rhr.to_frame("resting_hr")
            result["resting_hr_weekly"] = daily_rhr.resample("W").mean().round(1).to_frame("resting_hr")
        else:
            result["resting_hr_daily"] = pd.DataFrame()
            result["resting_hr_weekly"] = pd.DataFrame()

        # HRV
        hrv = self.data.get("hrv")
        if hrv is not None and not hrv.empty:
            daily_hrv = hrv.groupby(hrv.index.date)["value"].mean().rename_axis("date")
            daily_hrv.index = pd.to_datetime(daily_hrv.index)
            result["hrv_daily"] = daily_hrv.to_frame("hrv")
            result["hrv_weekly"] = daily_hrv.resample("W").mean().round(1).to_frame("hrv")
        else:
            result["hrv_daily"] = pd.DataFrame()
            result["hrv_weekly"] = pd.DataFrame()

        return result

    # ------------------------------------------------------------------
    # Activity
    # ------------------------------------------------------------------

    def activity_analysis(self) -> Dict[str, pd.DataFrame]:
        """Active energy burned, exercise minutes and stand hours."""
        result = {}

        for key, label in [
            ("active_energy", "active_kcal"),
            ("exercise_minutes", "exercise_min"),
        ]:
            df = self.data.get(key)
            if df is not None and not df.empty:
                daily = df.groupby(df.index.date)["value"].sum().rename_axis("date")
                daily.index = pd.to_datetime(daily.index)
                result[f"{label}_daily"] = daily.to_frame(label)
                result[f"{label}_weekly"] = daily.resample("W").mean().round(1).to_frame(label)
            else:
                result[f"{label}_daily"] = pd.DataFrame()
                result[f"{label}_weekly"] = pd.DataFrame()

        # Stand hours — category type, count per day
        stand = self.data.get("stand_hour")
        if stand is not None and not stand.empty:
            stood = stand[stand["value"] == "HKCategoryValueAppleStandHourStood"]
            daily_stand = stood.groupby(stood.index.date).size().rename_axis("date")
            daily_stand.index = pd.to_datetime(daily_stand.index)
            result["stand_hours_daily"] = daily_stand.to_frame("stand_hours")
        else:
            result["stand_hours_daily"] = pd.DataFrame()

        return result

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def noise_analysis(self) -> Dict[str, pd.DataFrame]:
        """Environmental noise exposure patterns."""
        df = self.data.get("noise_db")
        if df is None or df.empty:
            return {"daily": pd.DataFrame(), "hourly_avg": pd.DataFrame()}

        daily = df.groupby(df.index.date)["value"].agg(["mean", "max", "min"]).rename_axis("date")
        daily.index = pd.to_datetime(daily.index)
        daily.columns = ["avg_db", "max_db", "min_db"]

        hourly = df.groupby(df.index.hour)["value"].mean().rename_axis("hour")
        hourly = hourly.to_frame("avg_db").round(1)

        return {"daily": daily.round(1), "hourly_avg": hourly}

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def correlation_matrix(self) -> pd.DataFrame:
        """Cross-metric daily correlations."""
        parts = {}

        # Steps
        steps = self.data.get("steps")
        if steps is not None and not steps.empty:
            parts["steps"] = steps.groupby(steps.index.date)["value"].sum()

        # Resting HR
        rhr = self.data.get("resting_hr")
        if rhr is not None and not rhr.empty:
            parts["resting_hr"] = rhr.groupby(rhr.index.date)["value"].mean()

        # HRV
        hrv = self.data.get("hrv")
        if hrv is not None and not hrv.empty:
            parts["hrv"] = hrv.groupby(hrv.index.date)["value"].mean()

        # Active energy
        ae = self.data.get("active_energy")
        if ae is not None and not ae.empty:
            parts["active_kcal"] = ae.groupby(ae.index.date)["value"].sum()

        # Exercise minutes
        ex = self.data.get("exercise_minutes")
        if ex is not None and not ex.empty:
            parts["exercise_min"] = ex.groupby(ex.index.date)["value"].sum()

        # Noise
        noise = self.data.get("noise_db")
        if noise is not None and not noise.empty:
            parts["noise_db"] = noise.groupby(noise.index.date)["value"].mean()

        # Sleep total
        sleep = self.data.get("sleep")
        if sleep is not None and not sleep.empty:
            sleep_map = {
                "HKCategoryValueSleepAnalysisAsleepUnspecified": True,
                "HKCategoryValueSleepAnalysisAsleepCore": True,
                "HKCategoryValueSleepAnalysisAsleepDeep": True,
                "HKCategoryValueSleepAnalysisAsleepREM": True,
            }
            s = sleep.copy()
            s["is_sleep"] = s["value"].map(sleep_map).fillna(False)
            s = s[s["is_sleep"]]
            s["hours"] = (s["end"] - s.index).dt.total_seconds() / 3600
            s["night"] = s.index.normalize()
            mask = s.index.hour < 12
            s.loc[mask, "night"] = s.loc[mask, "night"] - pd.Timedelta(days=1)
            parts["sleep_hours"] = s.groupby(s["night"].dt.date)["hours"].sum()

        if len(parts) < 2:
            return pd.DataFrame()

        combined = pd.DataFrame(parts)
        return combined.corr().round(3)
