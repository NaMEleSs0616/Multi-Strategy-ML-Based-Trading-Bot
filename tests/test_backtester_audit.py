"""Look-ahead bias audit tests."""

from backtesting.event_driven_backtester import run_lookahead_bias_audit


def test_lookahead_bias_audit_passes():
    result = run_lookahead_bias_audit(n_bars=120, seed=42)
    assert result.passed, result.violations
