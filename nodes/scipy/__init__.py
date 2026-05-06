"""SciPy nodes — stats, signal, interp/fit (consolidated 10 → 3).

Active nodes:
- SpStatsNode  — describe | pearsonr | ttest | norm_pdf
- SpSignalNode — butter   | convolve | fft   | histogram
- SpFitNode    — interp1d | curve_fit

Back-compat aliases let the old class names keep importing — same class,
caller must set the appropriate `op` after construction.
"""
from nodes.scipy.sp_stats  import SpStatsNode
from nodes.scipy.sp_signal import SpSignalNode
from nodes.scipy.sp_fit    import SpFitNode

# Back-compat aliases
SpDescribeNode    = SpStatsNode    # op="describe"
SpNormPdfNode     = SpStatsNode    # op="norm_pdf"
SpTTestNode       = SpStatsNode    # op="ttest"
SpCorrelationNode = SpStatsNode    # op="pearsonr"

SpHistogramNode   = SpSignalNode   # op="histogram"
SpFFTNode         = SpSignalNode   # op="fft"
SpButterworthNode = SpSignalNode   # op="butter"
SpConvolveNode    = SpSignalNode   # op="convolve"

SpInterp1dNode    = SpFitNode      # op="interp1d"
SpCurveFitNode    = SpFitNode      # op="curve_fit"

__all__ = [
    "SpStatsNode", "SpSignalNode", "SpFitNode",
    "SpDescribeNode", "SpNormPdfNode", "SpTTestNode", "SpCorrelationNode",
    "SpHistogramNode", "SpFFTNode", "SpButterworthNode", "SpConvolveNode",
    "SpInterp1dNode", "SpCurveFitNode",
]
