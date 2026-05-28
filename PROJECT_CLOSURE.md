# shrimp-vision-router Closure Contract

## Summary

- source: explicit
- profile: python
- mode: build

## Boot

- ready_check.type: `http`

## Verification

- smoke_checks:
  - `service-smoke` (command)

## Critical Flows

- `main-flow`: Start the service and exercise the primary request path.

## Gates

- boot_ok: `False`
- tests_ok: `True`
- critical_flow_count: `1`
- max_open_regressions: `0`

## Autonomy

- repair_budget: `8`
- stall_threshold: `3`
- feature_freeze_on_red: `True`

## Assumptions

- none
