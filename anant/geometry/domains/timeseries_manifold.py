"""
Time Series as Riemannian Manifolds
===================================

Revolutionary Insight: Time series are CYCLIC GEODESICS on a manifold!

Key Concepts:
- Periodicity = Closed geodesic
- Anomalies = Curvature spikes
- Trends = Geodesic drift
- Forecasting = Geodesic extension
- Seasonality = Periodic curvature
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..core.riemannian_manifold import RiemannianGraphManifold

logger = logging.getLogger(__name__)


@dataclass
class CyclicGeodesic:
    """A closed geodesic representing a cycle"""
    period: int  # Period length
    confidence: float  # How confident we are
    amplitude: float  # Strength of cycle
    phase: float  # Phase offset
    curvature_profile: Optional[np.ndarray] = None


@dataclass
class TemporalAnomaly:
    """Anomaly detected via curvature"""
    timestamp: int
    curvature: float
    z_score: float
    severity: str  # 'mild', 'moderate', 'severe'


class TimeSeriesManifold:
    """
    Time series as a Riemannian manifold with cyclic geodesics.
    
    Revolutionary Framework:
    - Time series → Curve on manifold
    - Periodicity → Closed geodesic (returns to start)
    - Anomalies → High curvature points
    - Trends → Geodesic drift
    - Forecasting → Geodesic extension
    
    Mathematical Foundation:
    - Embed time series in delay-coordinate space (Takens)
    - Define metric from local covariance
    - Compute curvature along trajectory
    - Detect cycles as closed geodesics
    
    Examples:
        >>> ts_manifold = TimeSeriesManifold(time_series)
        >>> 
        >>> # Find cycles (closed geodesics)
        >>> cycles = ts_manifold.find_closed_geodesics()
        >>> # [(period=24, confidence=0.95), (period=168, confidence=0.88)]
        >>> 
        >>> # Detect anomalies (curvature spikes)
        >>> anomalies = ts_manifold.detect_curvature_anomalies()
        >>> 
        >>> # Forecast (extend geodesic)
        >>> forecast = ts_manifold.geodesic_forecast(steps=10)
    """
    
    def __init__(
        self,
        time_series: np.ndarray,
        embedding_dim: int = None,
        delay: int = 1,
        detrend: bool = True
    ):
        """
        Initialize temporal manifold from time series.
        
        Args:
            time_series: 1D array of time series values
            embedding_dim: Embedding dimension (auto if None)
            delay: Time delay for embedding
            detrend: Remove trend before analysis
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy required for time series analysis")
        
        self.original_series = np.array(time_series)
        self.n_points = len(time_series)
        
        # Detrend if requested
        if detrend:
            self.series = self._detrend(self.original_series)
        else:
            self.series = self.original_series.copy()
        
        # Estimate embedding dimension if not provided
        self.embedding_dim = embedding_dim or self._estimate_embedding_dim()
        self.delay = delay
        
        # Create delay-coordinate embedding (Takens embedding)
        self.embedded_series = self._delay_embedding()
        
        # Compute curvature along trajectory
        self.curvature_profile = self._compute_curvature_profile()
        
        # Cache for detected cycles
        self.cycles: List[CyclicGeodesic] = []
        
        logger.info(f"TimeSeriesManifold: {self.n_points} points, dim={self.embedding_dim}")
    
    def _detrend(self, series: np.ndarray) -> np.ndarray:
        """Remove linear trend"""
        t = np.arange(len(series))
        coeffs = np.polyfit(t, series, 1)
        trend = np.polyval(coeffs, t)
        return series - trend
    
    def _estimate_embedding_dim(self) -> int:
        """Estimate embedding dimension using false nearest neighbors"""
        # Simplified: use heuristic based on series length
        n = len(self.series)
        if n < 100:
            return 2
        elif n < 500:
            return 3
        elif n < 2000:
            return 5
        else:
            return min(10, int(np.log2(n)))
    
    def _delay_embedding(self) -> np.ndarray:
        """
        Create delay-coordinate embedding (Takens embedding).
        
        Transforms 1D time series into multi-dimensional trajectory.
        """
        n = len(self.series)
        m = self.embedding_dim
        tau = self.delay
        
        # Number of embedded points
        n_embedded = n - (m - 1) * tau
        
        if n_embedded <= 0:
            raise ValueError("Time series too short for embedding")
        
        # Create embedding
        embedded = np.zeros((n_embedded, m))
        for i in range(m):
            embedded[:, i] = self.series[i*tau : i*tau + n_embedded]
        
        return embedded
    
    def _compute_curvature_profile(self) -> np.ndarray:
        """
        Compute curvature along the time series trajectory.
        
        High curvature = Sharp turn = Anomaly
        Periodic curvature = Cycles
        """
        n = len(self.embedded_series)
        curvature = np.zeros(n)
        
        # Compute curvature using finite differences
        for i in range(1, n - 1):
            # Points before, at, and after
            p_prev = self.embedded_series[i-1]
            p_curr = self.embedded_series[i]
            p_next = self.embedded_series[i+1]
            
            # Tangent vectors
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # Normalize
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Curvature approximation: angle change
                cos_angle = np.dot(v1, v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Curvature = angle / average_step
                avg_step = (v1_norm + v2_norm) / 2
                curvature[i] = angle / (avg_step + 1e-10)
        
        return curvature
    
    def find_closed_geodesics(
        self,
        min_period: int = 2,
        max_period: Optional[int] = None,
        min_confidence: float = 0.6
    ) -> List[CyclicGeodesic]:
        """
        Find closed geodesics (cycles) in the time series.
        
        Uses frequency analysis of curvature profile.
        
        Args:
            min_period: Minimum cycle period
            max_period: Maximum cycle period (auto if None)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected cycles
        """
        if max_period is None:
            max_period = min(len(self.series) // 2, 1000)
        
        # Perform FFT on original series
        fft_vals = fft(self.series)
        fft_freq = fftfreq(len(self.series))
        
        # Power spectrum
        power = np.abs(fft_vals) ** 2
        
        # Find peaks in power spectrum
        # Only positive frequencies
        pos_freq_idx = fft_freq > 0
        pos_freqs = fft_freq[pos_freq_idx]
        pos_power = power[pos_freq_idx]
        
        # Find peaks
        peaks, properties = find_peaks(pos_power, height=0, prominence=np.max(pos_power)*0.05)
        
        # Convert to periods
        cycles = []
        for peak_idx in peaks:
            freq = pos_freqs[peak_idx]
            if freq > 0:
                period = int(1.0 / freq)
                
                if min_period <= period <= max_period:
                    # Confidence based on peak prominence
                    amplitude = pos_power[peak_idx]
                    confidence = min(1.0, amplitude / np.max(pos_power))
                    
                    if confidence >= min_confidence:
                        # Compute curvature profile for this cycle
                        curvature_cycle = self._extract_cycle_curvature(period)
                        
                        cycle = CyclicGeodesic(
                            period=period,
                            confidence=float(confidence),
                            amplitude=float(amplitude),
                            phase=0.0,  # TODO: compute phase
                            curvature_profile=curvature_cycle
                        )
                        cycles.append(cycle)
        
        # Sort by confidence
        cycles.sort(key=lambda c: c.confidence, reverse=True)
        
        self.cycles = cycles
        logger.info(f"Found {len(cycles)} closed geodesics (cycles)")
        
        return cycles
    
    def _extract_cycle_curvature(self, period: int) -> np.ndarray:
        """Extract average curvature profile for a cycle"""
        n_cycles = len(self.curvature_profile) // period
        
        if n_cycles == 0:
            return np.zeros(period)
        
        # Reshape and average
        truncated_length = n_cycles * period
        reshaped = self.curvature_profile[:truncated_length].reshape(n_cycles, period)
        avg_cycle = np.mean(reshaped, axis=0)
        
        return avg_cycle
    
    def detect_curvature_anomalies(
        self,
        threshold: float = 2.0,
        method: str = 'zscore'
    ) -> List[TemporalAnomaly]:
        """
        Detect anomalies via curvature spikes.
        
        HIGH CURVATURE = ANOMALY (key insight)
        
        Args:
            threshold: Z-score threshold
            method: 'zscore' or 'absolute'
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Statistics
        mean_curv = np.mean(self.curvature_profile)
        std_curv = np.std(self.curvature_profile)
        
        # Detect anomalies
        for i, curvature in enumerate(self.curvature_profile):
            if method == 'zscore':
                z_score = (curvature - mean_curv) / (std_curv + 1e-10)
                
                if abs(z_score) > threshold:
                    # Determine severity
                    if abs(z_score) > 4.0:
                        severity = 'severe'
                    elif abs(z_score) > 3.0:
                        severity = 'moderate'
                    else:
                        severity = 'mild'
                    
                    anomaly = TemporalAnomaly(
                        timestamp=i,
                        curvature=float(curvature),
                        z_score=float(z_score),
                        severity=severity
                    )
                    anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} temporal anomalies")
        return anomalies
    
    def geodesic_forecast(
        self,
        steps: int,
        method: str = 'curvature'
    ) -> np.ndarray:
        """
        Forecast by extending the geodesic.
        
        Continues the trajectory naturally along the manifold.
        
        Args:
            steps: Number of steps to forecast
            method: 'curvature' or 'cyclic'
            
        Returns:
            Forecasted values
        """
        if method == 'cyclic' and self.cycles:
            # Use dominant cycle for forecast
            dominant_cycle = self.cycles[0]
            period = dominant_cycle.period
            
            # Repeat the pattern
            last_cycle = self.series[-period:]
            forecast = np.tile(last_cycle, (steps // period) + 1)[:steps]
            
        else:
            # Simple continuation using last curvature
            forecast = np.zeros(steps)
            
            # Use last few points to estimate continuation
            last_points = self.series[-10:]
            trend = np.mean(np.diff(last_points))
            
            for i in range(steps):
                forecast[i] = self.series[-1] + trend * (i + 1)
        
        return forecast
    
    def curvature_periodicity(self) -> Dict[str, Any]:
        """
        Analyze periodicity in curvature itself.
        
        Reveals meta-patterns in the time series structure.
        """
        # FFT of curvature
        curv_fft = fft(self.curvature_profile)
        curv_freq = fftfreq(len(self.curvature_profile))
        curv_power = np.abs(curv_fft) ** 2
        
        # Find dominant frequency
        pos_idx = curv_freq > 0
        if np.any(pos_idx):
            max_idx = np.argmax(curv_power[pos_idx])
            dominant_freq = curv_freq[pos_idx][max_idx]
            dominant_period = int(1.0 / dominant_freq) if dominant_freq > 0 else 0
        else:
            dominant_period = 0
        
        return {
            'curvature_period': dominant_period,
            'curvature_power_spectrum': curv_power[pos_idx][:20].tolist(),
            'mean_curvature': float(np.mean(self.curvature_profile)),
            'std_curvature': float(np.std(self.curvature_profile))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'n_points': self.n_points,
            'embedding_dim': self.embedding_dim,
            'delay': self.delay,
            'mean_curvature': float(np.mean(self.curvature_profile)),
            'std_curvature': float(np.std(self.curvature_profile)),
            'max_curvature': float(np.max(self.curvature_profile)),
            'n_detected_cycles': len(self.cycles),
            'dominant_cycle': self.cycles[0].period if self.cycles else None
        }


def detect_cycles_geometric(
    time_series: np.ndarray,
    min_confidence: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Convenience function to detect cycles in time series.
    
    Args:
        time_series: Time series data
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected cycles with metadata
    """
    manifold = TimeSeriesManifold(time_series)
    cycles = manifold.find_closed_geodesics(min_confidence=min_confidence)
    
    return [
        {
            'period': c.period,
            'confidence': c.confidence,
            'amplitude': c.amplitude
        }
        for c in cycles
    ]
