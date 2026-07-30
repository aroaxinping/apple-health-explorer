"""
apple-health-explorer — Comprehensive analysis of Apple Health export data.
"""

__version__ = "0.1.0"

from .parser import HealthParser
from .analyzer import HealthAnalyzer
from .visualizer import HealthVisualizer


class HealthExport:
    """Main entry point. Wraps parser, analyzer and visualizer."""

    def __init__(self, path: str):
        self.parser = HealthParser(path)
        self.data = self.parser.parse()
        self._analyzer = HealthAnalyzer(self.data)
        self._visualizer = HealthVisualizer(self.data)

    # --- Analysis ---

    def steps_analysis(self):
        """Daily/weekly/monthly step patterns and trends."""
        return self._analyzer.steps_analysis()

    def sleep_analysis(self):
        """Sleep duration, consistency and quality metrics."""
        return self._analyzer.sleep_analysis()

    def heart_analysis(self):
        """Resting heart rate trends and HRV patterns."""
        return self._analyzer.heart_analysis()

    def activity_analysis(self):
        """Active energy, exercise minutes, stand hours."""
        return self._analyzer.activity_analysis()

    def noise_analysis(self):
        """Environmental noise exposure patterns."""
        return self._analyzer.noise_analysis()

    def correlation_matrix(self):
        """Cross-metric correlations (sleep vs steps, HRV vs activity, etc.)."""
        return self._analyzer.correlation_matrix()

    # --- Visualization ---

    def plot_steps(self, ax=None):
        return self._visualizer.plot_steps(ax)

    def plot_sleep(self, ax=None):
        return self._visualizer.plot_sleep(ax)

    def plot_heart(self, ax=None):
        return self._visualizer.plot_heart(ax)

    def plot_correlation_matrix(self, ax=None):
        return self._visualizer.plot_correlation_matrix(ax)
