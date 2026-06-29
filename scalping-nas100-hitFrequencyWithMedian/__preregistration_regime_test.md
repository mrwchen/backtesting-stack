# Pre-Registration — Cross-Regime Robustness Test (Phase 3)

Written BEFORE running, to prevent meta-overfitting (tuning against the OOS
result). The run is executed exactly once with the config below. The verdict is
read off the pre-committed rule. We do NOT re-tune to make a failing year pass.

## Question (the only one this run answers)

Is the global walk-forward METHOD regime-robust — i.e. does its out-of-sample
performance hold across two different yearly regimes (2025 and 2026), or does the
edge exist only in the 2026 regime it was derived from?

This tests the method, not a single parameter set.

## Fixed config (no changes after this point)

- RUN_MODE = walk_forward, global mode (WF_GLOBAL_PARAMETER_SET default = true)
- Period: 2025-01-01 .. latest tick (full 2025 + 2026-H1)
- All 13 sessions ENABLED (revert the exploratory 2-session pruning — that choice
  was itself derived from a 2026 OOS result and would compound researcher bias)
- Costs: SLIPPAGE_POINTS = 1.0, COMMISSION_PER_UNIT = 0.0 (Pepperstone indices)
- WF matrix: TRAIN 20,40,60 x TEST 10,20 (6 combos), Stage 2 on
- Seeds fixed (12345)

## Acceptance rule (committed before seeing results)

A PASS requires ALL of:

1. **Cross-year:** split OOS folds by calendar year of the OOS window. BOTH the
   2025 OOS folds AND the 2026 OOS folds must be net positive with aggregate
   PF > 1.1, in the best-by-design combo. No single year may carry the result.
2. **Breadth:** at least 4 of the 6 window combos must be OOS net-positive overall
   (Stage 2). Not a single cherry-picked combo.
3. **Drawdown:** realized OOS max drawdown <= 35% (the 2025+2026 static run hit
   -60%; that is the red flag we are gating against).
4. **Trades:** >= 200 OOS trades per year so the per-year verdict is meaningful.

## If it FAILS

Verdict = the strategy is NOT regime-robust as-is. The legitimate next move is a
regime filter designed from market logic (e.g. volatility/trend state), NOT a
parameter search that happens to make 2025 pass. Do not iterate the matrix or
sessions against this result.

## Honesty note

2025 monthly P&L has already been observed once (run 25 year-split), so 2025 is no
longer a pristine holdout. This pre-registration constrains future degrees of
freedom from here on; the only fully-clean test left is genuinely future data.
