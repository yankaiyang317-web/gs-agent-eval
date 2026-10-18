This repository is an offline evaluator, separate from gs-agent navigation.
Never feed GT, reliability states or evaluation scores into an acting Agent.
Do not change frozen profiles, states, assets or formulas to favor a tested run.
Create a new benchmark_id for protocol/state changes; a new evaluator version for code changes.
Keep A coverage and B utility separate; do not introduce an overall score.
Do not mutate source PLYs or recorded trajectories; write new output directories.
Research snapshots are archived scene-specific recipes, not the daily scoring CLI.
Run CPU tests and fixed-input parity before declaring a scoring change equivalent.
