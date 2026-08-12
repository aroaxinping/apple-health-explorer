"""
Tests for the Apple Health export parser using synthetic XML data.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from apple_health_explorer.parser import HealthParser


SYNTHETIC_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
]>
<HealthData locale="es_ES">
 <ExportDate value="2026-04-01 10:00:00 +0200"/>

 <Record type="HKQuantityTypeIdentifierStepCount"
         sourceName="iPhone" unit="count"
         startDate="2026-03-01 08:00:00 +0100"
         endDate="2026-03-01 08:15:00 +0100"
         value="120"/>
 <Record type="HKQuantityTypeIdentifierStepCount"
         sourceName="iPhone" unit="count"
         startDate="2026-03-01 12:00:00 +0100"
         endDate="2026-03-01 12:30:00 +0100"
         value="3500"/>
 <Record type="HKQuantityTypeIdentifierStepCount"
         sourceName="iPhone" unit="count"
         startDate="2026-03-02 09:00:00 +0100"
         endDate="2026-03-02 09:20:00 +0100"
         value="2000"/>

 <Record type="HKQuantityTypeIdentifierRestingHeartRate"
         sourceName="Apple Watch" unit="count/min"
         startDate="2026-03-01 07:00:00 +0100"
         endDate="2026-03-01 07:00:00 +0100"
         value="62"/>
 <Record type="HKQuantityTypeIdentifierRestingHeartRate"
         sourceName="Apple Watch" unit="count/min"
         startDate="2026-03-02 07:00:00 +0100"
         endDate="2026-03-02 07:00:00 +0100"
         value="58"/>

 <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
         sourceName="Apple Watch" unit="ms"
         startDate="2026-03-01 23:30:00 +0100"
         endDate="2026-03-01 23:30:00 +0100"
         value="45"/>

 <Record type="HKQuantityTypeIdentifierActiveEnergyBurned"
         sourceName="Apple Watch" unit="kcal"
         startDate="2026-03-01 10:00:00 +0100"
         endDate="2026-03-01 10:30:00 +0100"
         value="150"/>

 <Record type="HKQuantityTypeIdentifierAppleExerciseTime"
         sourceName="Apple Watch" unit="min"
         startDate="2026-03-01 10:00:00 +0100"
         endDate="2026-03-01 10:30:00 +0100"
         value="30"/>

 <Record type="HKQuantityTypeIdentifierEnvironmentalAudioExposure"
         sourceName="Apple Watch" unit="dBASPL"
         startDate="2026-03-01 14:00:00 +0100"
         endDate="2026-03-01 14:30:00 +0100"
         value="72.5"/>

 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
         sourceName="Apple Watch"
         startDate="2026-03-01 00:00:00 +0100"
         endDate="2026-03-01 02:00:00 +0100"
         value="HKCategoryValueSleepAnalysisAsleepCore"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
         sourceName="Apple Watch"
         startDate="2026-03-01 02:00:00 +0100"
         endDate="2026-03-01 03:00:00 +0100"
         value="HKCategoryValueSleepAnalysisAsleepDeep"/>
 <Record type="HKCategoryTypeIdentifierSleepAnalysis"
         sourceName="Apple Watch"
         startDate="2026-03-01 03:00:00 +0100"
         endDate="2026-03-01 03:30:00 +0100"
         value="HKCategoryValueSleepAnalysisAsleepREM"/>

 <Record type="HKCategoryTypeIdentifierAppleStandHour"
         sourceName="Apple Watch"
         startDate="2026-03-01 09:00:00 +0100"
         endDate="2026-03-01 10:00:00 +0100"
         value="HKCategoryValueAppleStandHourStood"/>
 <Record type="HKCategoryTypeIdentifierAppleStandHour"
         sourceName="Apple Watch"
         startDate="2026-03-01 10:00:00 +0100"
         endDate="2026-03-01 11:00:00 +0100"
         value="HKCategoryValueAppleStandHourIdle"/>

</HealthData>
"""


@pytest.fixture
def parsed_data(tmp_path):
    xml_file = tmp_path / "export.xml"
    xml_file.write_text(SYNTHETIC_XML)
    parser = HealthParser(str(xml_file))
    return parser.parse()


class TestParserBasic:
    def test_returns_dict_with_expected_keys(self, parsed_data):
        expected = {"steps", "resting_hr", "hrv", "active_energy", "exercise_minutes",
                    "noise_db", "heart_rate", "sleep", "stand_hour"}
        assert expected == set(parsed_data.keys())

    def test_steps_count(self, parsed_data):
        assert len(parsed_data["steps"]) == 3

    def test_steps_values_are_numeric(self, parsed_data):
        assert parsed_data["steps"]["value"].dtype in ("float64", "int64")

    def test_steps_total(self, parsed_data):
        assert parsed_data["steps"]["value"].sum() == 5620

    def test_resting_hr_count(self, parsed_data):
        assert len(parsed_data["resting_hr"]) == 2

    def test_hrv_count(self, parsed_data):
        assert len(parsed_data["hrv"]) == 1
        assert parsed_data["hrv"]["value"].iloc[0] == 45.0

    def test_sleep_records(self, parsed_data):
        assert len(parsed_data["sleep"]) == 3

    def test_sleep_values_are_strings(self, parsed_data):
        assert parsed_data["sleep"]["value"].dtype == object

    def test_active_energy(self, parsed_data):
        assert parsed_data["active_energy"]["value"].iloc[0] == 150.0

    def test_exercise_minutes(self, parsed_data):
        assert parsed_data["exercise_minutes"]["value"].iloc[0] == 30.0

    def test_noise(self, parsed_data):
        assert parsed_data["noise_db"]["value"].iloc[0] == 72.5

    def test_stand_hour(self, parsed_data):
        assert len(parsed_data["stand_hour"]) == 2

    def test_datetime_index(self, parsed_data):
        for name, df in parsed_data.items():
            if not df.empty:
                assert isinstance(df.index, pd.DatetimeIndex), f"{name} should have DatetimeIndex"

    def test_empty_metrics_return_empty_df(self, parsed_data):
        # heart_rate has no records in our synthetic data
        assert parsed_data["heart_rate"].empty


class TestParserEdgeCases:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            HealthParser("/nonexistent/export.xml")

    def test_empty_xml(self, tmp_path):
        xml_file = tmp_path / "empty.xml"
        xml_file.write_text('<?xml version="1.0"?><HealthData></HealthData>')
        data = HealthParser(str(xml_file)).parse()
        for df in data.values():
            assert df.empty
