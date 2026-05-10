# Pipeline driver

The orchestrator for the full run: spawns the environmental and
Sentinel-2 → PaddockTS pipelines on parallel threads, displays a live
status dashboard, and captures all output (Python, logging, warnings,
and C-level stderr) into a bounded ring buffer for safe rendering.

::: PaddockTS.get_outputs.get_outputs
