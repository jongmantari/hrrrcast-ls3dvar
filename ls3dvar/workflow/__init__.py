"""Workflow helpers for the experiment drivers (cli/run_cycles.py, cli/run_forecasts.py).

Pure orchestration: nothing here imports TensorFlow. Every GPU step runs as its own
process (timeout + retry, deterministic launch), exactly as scripts/cycle_ls3dvar.sh does.
"""
