"""
Parse Apple Health export.xml into clean DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from lxml import etree


# HKQuantityTypeIdentifier records we care about
QUANTITY_TYPES = {
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
    "HKQuantityTypeIdentifierAppleExerciseTime": "exercise_minutes",
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure": "noise_db",
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
}

# HKCategoryTypeIdentifier records
CATEGORY_TYPES = {
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    "HKCategoryTypeIdentifierAppleStandHour": "stand_hour",
}


class HealthParser:
    """Parse an Apple Health export.xml file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Export file not found: {self.path}")

    def parse(self) -> Dict[str, pd.DataFrame]:
        """Parse the XML and return a dict of DataFrames keyed by metric name."""
        records: Dict[str, list] = {v: [] for v in {**QUANTITY_TYPES, **CATEGORY_TYPES}.values()}

        context = etree.iterparse(str(self.path), events=("end",), tag="Record")

        for _, elem in context:
            rtype = elem.get("type")

            if rtype in QUANTITY_TYPES:
                name = QUANTITY_TYPES[rtype]
                records[name].append(self._parse_quantity(elem))

            elif rtype in CATEGORY_TYPES:
                name = CATEGORY_TYPES[rtype]
                records[name].append(self._parse_category(elem))

            # Free memory — critical for large XML files
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        return {name: self._to_dataframe(rows, name) for name, rows in records.items()}

    @staticmethod
    def _parse_quantity(elem) -> dict:
        return {
            "start": elem.get("startDate"),
            "end": elem.get("endDate"),
            "value": float(elem.get("value", 0)),
            "source": elem.get("sourceName", ""),
            "unit": elem.get("unit", ""),
        }

    @staticmethod
    def _parse_category(elem) -> dict:
        return {
            "start": elem.get("startDate"),
            "end": elem.get("endDate"),
            "value": elem.get("value", ""),
            "source": elem.get("sourceName", ""),
        }

    @staticmethod
    def _to_dataframe(rows: list, name: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Parse dates
        for col in ("start", "end"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed", utc=False)

        # Set start as index
        if "start" in df.columns:
            df = df.set_index("start").sort_index()

        # Ensure numeric value for quantity types
        if "value" in df.columns and name not in ("sleep", "stand_hour"):
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df
