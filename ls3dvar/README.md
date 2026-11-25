# ls3dvar: stand-alone LS-3DVar package

`ls3dvar` contains only what the analysis driver and the two parity tests need. It
**never imports `mantari`**, and it lives next to `mantari/` and `src/` in the same
repository:

```
<repo>/
  ls3dvar/          this package (new, maintained)
  mantari/          old code: untouched, kept as the reference until the switch-over
  src/              HRRRCast source (resnet, fcst, diffusion_params)
  net-diffusion/model.keras
```

This first version is a **mechanical copy**. Every module is the original source,
with only its import and path lines changed. The one exception is
`analysis/diagnostics.py`, which carries the convergence-plot fix (it adds PNG
files and no numbers change). `tools/extract/EXTRACTION.diff` lists every changed line;
`tools/extract/extract_ls3dvar.py` regenerates the package from the original sources and
refuses to finish if any `mantari` import is left.

## Tools (inside the package)

| | |
|---|---|
| `python -m ls3dvar.tools.gate` | parity gate: `snapshot`, `compare [--use ls3dvar]`, `diffdirs OLD NEW` |
| `python -m ls3dvar.tools.inventory` | read-only import map of the old `mantari/` tree |
| `tools/extract/` | how this package was made: `extract_ls3dvar.py`, `diagnostics_fixed.py`, `EXTRACTION.diff` |

`tools/extract/extract_ls3dvar.py` is the **one-time bootstrap**. It refuses to write into an
existing `ls3dvar/` (pass `--force` to override): once you edit the package, regenerating it
would overwrite your edits. To inspect a regeneration, write it to an empty folder and
`diff -r` the two. The gate writes its results to `gate/` at the repo root; add that to
`.gitignore`.

## Running

From the repository root (as before):

```bash
python -m ls3dvar.cli.run_analysis --config case.yaml [same options as before]
python -m ls3dvar.tests.test_decode_parity
python -m ls3dvar.tests.test_encode_decode_roundtrip --anchor
python -m ls3dvar.cli.make_forecast_background | large_scale_blend | verify_state | pool_station_reports ...
```

The command-line options, the config keys, `$MANTARI_CONFIG`, the default config
(`mantari/config/test.yaml`: only the YAML file is read) and the output file
names are all the same as the old driver's.

## Where things came from

| ls3dvar | was |
|---|---|
| `cli/run_analysis.py` | `mantari/workflows/test_ls3dvar_analysis_t2m_d2m_ps.py` |
| `cli/make_forecast_background.py` | `mantari/workflows/make_forecast_background.py` |
| `cli/large_scale_blend.py` | `mantari/ls3dvar/large_scale_blend.py` (the default `normalize-stats.nc` still resolves to `<repo>/net-diffusion/`) |
| `cli/verify_state.py` | `mantari/workflows/verify_state.py` |
| `cli/pool_station_reports.py` | `mantari/workflows/pool_station_reports.py` |
| `scripts/cycle_ls3dvar.sh` | `cycle_ls3dvar.sh` + `PKG` switch (below) |
| `tests/test_decode_parity.py` | `mantari/workflows/test_decode_parity.py` |
| `tests/test_encode_decode_roundtrip.py` | `mantari/workflows/test_encode_decode_roundtrip.py` |
| `paths.py` | the `mantari/common.py` constants that are used (`REPO_ROOT`, `SRC_DIR`, `MODEL_PATH`, `DEFAULT_CONFIG` = old `TEST_CONFIG`) |
| **hrrrcast/** network pieces | |
| `hrrrcast/encoder.py` | `mantari/encoder_runtime.py` |
| `hrrrcast/processor.py` | `mantari/processor_runtime_tf.py` |
| `hrrrcast/decoder.py` | `mantari/decoder_runtime_tf.py` (differentiable; used by the analysis) |
| `hrrrcast/decoder_numpy.py` | `mantari/decoder_runtime.py` (numpy outputs; used by the parity tests) |
| `hrrrcast/tail.py` | `mantari/tail_model_runtime_tf.py` |
| `hrrrcast/latent.py` | `mantari/ls3dvar/latent_extractor.py` |
| `hrrrcast/analysis_input.py` | `mantari/ls3dvar/analysis_input_builder.py` |
| `hrrrcast/time_features.py` | `mantari/ls3dvar/time_features.py` |
| `hrrrcast/channels.py` | `mantari/ls3dvar/channels.py` |
| **analysis/** core | |
| `analysis/state_map.py` | `mantari/ls3dvar/analysis_state_runtime_tf.py` |
| `analysis/bmatrix.py` | `mantari/ls3dvar` (state blocks only) |
| `analysis/solver.py` | `mantari/ls3dvar/solver_multi.py` |
| `analysis/diagnostics.py` | `mantari/ls3dvar/diagnostics.py` (+ fix) |
| **obs/** observations | |
| `obs/types.py` | `mantari/observations.py` |
| `obs/ioda.py` | `mantari/observation_provider.py` |
| `obs/grid.py` | `mantari/station_mapper.py` |
| `obs/time_window.py` | `mantari/ls3dvar/observation_time.py` |
| `obs/pipeline.py` | `mantari/ls3dvar/observation_pipeline.py` |
| `obs/gross_filter.py` | `mantari/ls3dvar/observation_filter.py` |
| `obs/qc.py` | `mantari/ls3dvar/obs_qc.py` |
| `obs/holdout.py` | `mantari/ls3dvar/station_split.py` |
| `obs/operators.py` | `mantari/ls3dvar/operators.py` |

Class and function names did not change.

## Switching over safely

```bash
# 1. baseline from the OLD code (skip if gate/base already exists)
python -m ls3dvar.tools.gate snapshot --name base --driver-args "<your explicit settings>"
python -m ls3dvar.tools.gate compare --baseline base --name rerun          # determinism

# 2. the NEW package against that same baseline
python -m ls3dvar.tools.gate compare --baseline base --use ls3dvar --name pkg_v0
```

Expected result for step 2: **GATE PASS**, plus `info new output …/cost_convergence_outer_00.png`
lines (the diagnostics fix). Any `FAIL` means that the copy differs from the original:
send `gate/pkg_v0/compare.json` and the logs.

After that:

* Make changes **only in `ls3dvar/`**. After each one, run
  `compare --baseline base --use ls3dvar`.
* For a change that is meant to alter results: review the differences, then take a new
  baseline from the package: `snapshot --name base_v2 --use ls3dvar --driver-args "..."`.
* Once you trust the package, freeze `mantari/` (or move it to `legacy/`). Nothing in
  `ls3dvar/` depends on it.

## Cycling

`scripts/cycle_ls3dvar.sh` is your `cycle_ls3dvar.sh` with a single change: the four
modules it calls are chosen by `PKG`.

| `PKG` | analysis | background / forecasts | blend | verification |
|---|---|---|---|---|
| `ls3dvar` (default) | `ls3dvar.cli.run_analysis` | `ls3dvar.cli.make_forecast_background` | `ls3dvar.cli.large_scale_blend` | `ls3dvar.cli.verify_state` |
| `mantari` | `mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps` | `mantari.workflows.make_forecast_background` | `mantari.ls3dvar.large_scale_blend` | `mantari.workflows.verify_state` |

Everything else (the config `cycling:` section, the environment overrides, `RESUME`,
hold-out, `VERIFY_QC`, the blend, the summaries) is unchanged. Each experiment
records which code produced it in `$EXP/code_version.txt`.

The default is plain TF (`DETERMINISTIC=0`, `experiment.deterministic: false`).
`DETERMINISTIC=1` makes every step bit-reproducible (about 1.5x slower): it sets
`TF_DETERMINISTIC_OPS=1` and launches each module through `tools/det_runner.py`, which
replaces the decoder's exact-2x nearest up-sampling by `tf.repeat` (identical forward,
deterministic gradient). Without it, two runs of the same cycle differ: on the
development GPU, 10 L-BFGS iterations turned a 5e-5 forward difference into 0.3 sigma
in x_a and 0.4 points in the improvement scores.

Cycling parity check: run the same short cycle with both, deterministically, and compare every npz and csv:

```bash
DETERMINISTIC=1 PKG=mantari EXP=$DATA/experiments/parity_mantari ls3dvar/scripts/cycle_ls3dvar.sh 2026-08-11T00 2026-08-11T02 cfg.yaml
DETERMINISTIC=1 PKG=ls3dvar EXP=$DATA/experiments/parity_ls3dvar ls3dvar/scripts/cycle_ls3dvar.sh 2026-08-11T00 2026-08-11T02 cfg.yaml
python -m ls3dvar.tools.gate diffdirs $DATA/experiments/parity_mantari $DATA/experiments/parity_ls3dvar
```

Expect `DIFFDIRS PASS` with no tolerance (gate status 2026-11-17: driver and parity tests bit-identical, old vs new).
`diffdirs` ignores the `path` column of `verification.csv` (it holds the experiment
folder) and only lists logs, YAML and PNG files.

The cycle script still uses the old default paths (`BASE_CFG`, `cycling.repo`). Run it
from anywhere; it `cd`s to `cycling.repo`, which must be the repo that contains `ls3dvar/`.

## Experiments in Python: `cli/run_cycles.py` + `cli/run_forecasts.py`

One YAML per experiment (every key, with its default: `configs/experiment_example.yaml`):

```bash
# start from an existing config (its cycling: section); then edit name, period, hold-out, ...
python -m ls3dvar.cli.run_cycles init --from $DATA/configs/frozen_v1/test.yaml \
    --name frozen_v1_py --start 2026-08-11T00 --end 2026-08-11T03 --out $DATA/exp/frozen_v1_py.yaml

python -m ls3dvar.cli.run_cycles check   $DATA/exp/frozen_v1_py.yaml   # settings, inputs, code, GPU
python -m ls3dvar.cli.run_cycles run     $DATA/exp/frozen_v1_py.yaml --detach
python -m ls3dvar.cli.run_cycles status | tail | stop | results  $DATA/exp/frozen_v1_py.yaml
```

Repeating an experiment of the old bash scripts:

```bash
python -m ls3dvar.cli.run_cycles import-old $DATA/experiments/c0812_mrms_assim_fine \
    --out $DATA/exp/c0812_mrms_assim_fine_py.yaml          # new name: c0812_mrms_assim_fine_py
python -m ls3dvar.cli.run_cycles results $DATA/experiments/c0812_mrms_assim_fine   # old folder: summary
```

`import-old` uses the old folder's `base_config.yaml` (the exact config that run used),
and takes the period, leads, hold-out, verify QC, extra analysis flags, blend and data
folders from the run log `$DATA/<name>.out`. Without a log it uses the cycle folders for
the period and the copied `run_experiment.sh` for the rest. Values that were passed
through the environment at launch are not in that copy, so the YAML notes which source
each value came from. The old folder is not touched.

| YAML section | What it controls |
|---|---|
| `experiment` | name, period, output folder, resume, `code` (ls3dvar \| mantari), `deterministic` |
| `paths` | data, reference_dir, obsdb_root, ...; usable anywhere as `${name}` |
| `background` | first cycle and later cycles: `reference_forecast` \| `cycled` \| `forecast_dir`; lead; GFS blend |
| `observations` | obsdb types fed to the analysis; station hold-out |
| `analysis` | on/off; `base_config` + `overrides` (deep-merged: any `ls3dvar:` / `observations:` key); extra flags |
| `forecast` | on/off; leads (1..6 h); from `an` and/or `bg` |
| `verification` | on/off; QC; one scoring config for experiments you compare |
| `runtime` | step timeout, analysis timeout, retries, live progress, environment |

Precomputed forecasts as backgrounds:

```bash
# F01..F03 from every reference HRRR analysis, 2026-08-10T23 .. 2026-08-11T02 (one model load per init)
python -m ls3dvar.cli.run_forecasts --start 2026-08-10T23 --end 2026-08-11T02 --length 3 \
    --reference-dir $DATA/reference/hrrr_ana --out $DATA/forecasts/ref_f03
#   -> $DATA/forecasts/ref_f03/<init>/fcst_f01.npz ... fcst_f03.npz  (+ forecast_manifest.yaml)

# in the experiment YAML:
#   background: {first_cycle: forecast_dir, source: forecast_dir,     # no cycling: every cycle
#                forecast_dir: ${data}/forecasts/ref_f03, lead: 1}    #   from the forecasts
#   background: {first_cycle: forecast_dir, source: cycled, ...}      # cycling after the first
```

`--init-from <experiment folder>` makes forecasts from that experiment's analyses instead.
`--length` is limited to 6 h, the range in which `make_forecast_background` reproduces
`fcst.py`. With `--init-from reference`, F01 is the same file as the cycle's cold-start
background (same tool and inputs, deterministic), so one forecast folder serves any
number of experiments.

Behaviour:

* **Same steps as `scripts/cycle_ls3dvar.sh`.** In a mock run, a fresh run made the same
  calls with the same arguments in the same order, wrote identical npz/CSV files and
  identical per-cycle configs, and printed the same summary. The bash scripts stay as
  the reference until a real parity run (`diffdirs`) passes.
* **One process per step**, with a timeout (the process group is killed) and retries
  for the analysis and forecasts, as in the bash cycle. `stop` stops only this experiment.
* **Resume** skips finished steps, including the verification of finished cycles (the
  bash cycle appended those rows again). It refuses an experiment folder made with
  other settings (also when `base_config` was edited in place), other code or another
  `deterministic` setting. Changing `start`, `end` or `runtime` is allowed, so a run
  can be extended.
* **Unknown YAML keys are an error.**
* **Cycling always makes the F01 it needs**, also with `forecast.enabled: false`, or
  with leads that leave out 1. (The bash cycle silently fell back to a cold start then.)

## End to end: `run_cycles pipeline` (prep -> cycles -> post)

One YAML runs the whole experiment, using the data-directory scripts as they are
(`workflow/stages.py` only picks the hours, skips what exists, writes the plot configs):

```bash
python -m ls3dvar.cli.run_cycles check    exp.yaml      # + which prep hours are missing, script imports
python -m ls3dvar.cli.run_cycles prep     exp.yaml --dry-run
python -m ls3dvar.cli.run_cycles pipeline exp.yaml --detach
python -m ls3dvar.cli.run_cycles post     exp.yaml --only diagnostics threat   # re-plot only
python -m ls3dvar.cli.run_cycles push     exp.yaml --dry-run                   # what would upload
```

| stage | script | hours / output |
|---|---|---|
| prep reference | `stage_reference.sh` (runs `run_cycle.sh` for hours never staged) | start-1 .. end+max lead, `<reference_dir>` |
| prep obs | `build_obsdb_cycles.py obsdb_cycles.yaml` | start .. end+max lead, `<obsdb_root>/<type>/<cycle>` |
| prep mrms | `mrms_stage.py` (`--grid-npz` from any staged reference) | start .. end+max lead, `prep.mrms.dir` |
| prep background | `run_forecasts` (only `background.source: forecast_dir`) | init start-lead .. end-lead |
| cycles | as `run` | `<exp>/<cycle>/` |
| post plots | `plot_ls3dvar_cycling.py --only verification precip` | `<exp>/figures/` |
| post fields | `plot_ls3dvar_cycling.py` (compare panels, MRMS column) | `<exp>/figures/fields/` |
| post diagnostics | `ls3dvar_diagnostics.py` (`<name>@bg` vs `<name>`) | `<exp>/figures/da_impact/` + `summary.md` |
| post threat | `plot_mrms_threat.py` | `<exp>/figures/da_impact/` |
| push | `aws s3 sync` (credentials + bucket checked first) | `paths.data` (or `<exp>`) -> `push.dest` |

`prep.reference.copy: true` replaces the links `stage_reference.sh` makes into `subrun/` by
copies, so `paths.data` is self-contained (what `push.what: all` uploads).
A missing hour that only verification needs is a warning; one the cycles need stops the
run before any cycle. Post configs land in `<exp>/post/*.yaml` (template + experiment paths,
cycles, leads + `set:`), logs in `<exp>/prep/` and `<exp>/post/`. Keys: `configs/experiment_example.yaml`.
Inside `{ }` YAML flow mappings, quote values with `${...}`: `{dir: "${fresh}/mrms"}`.

## Running experiments: `scripts/run_experiment.sh`

The package version of `run_experiment.sh`. It has the same commands and settings, plus
`PKG`, `DETERMINISTIC` (default 1) and `diff`. Copy it per experiment, as before:

```bash
cp $REPO/ls3dvar/scripts/run_experiment.sh $DATA/run_t2m06.sh
# edit: NAME=t2m06, CONFIG=configs/t2m06.yaml, VERIFY_CONFIG=$DATA/configs/frozen_v1/test.yaml
./run_t2m06.sh check && ./run_t2m06.sh run
./run_t2m06.sh status | tail | results | plot | stop | resume
./run_t2m06.sh diff frozen_v1_pkg          # every npz / csv against another experiment
```

Differences from the old script:

* It runs `$REPO/ls3dvar/scripts/cycle_ls3dvar.sh` (no copy in `$DATA` needed).
* `DETERMINISTIC=1` by default: experiments differ only by their settings, not by GPU
  noise (which reached 0.4 points of improvement between identical runs).
* `running` / `stop` / `status` also recognize package steps (`ls3dvar.cli...`,
  `det_runner.py`). Only bash/python processes match, so an editor or `less` on these
  files is never killed.
* Each experiment records its code (`code_version.txt`). `run` / `resume` refuse to
  continue an experiment made by other code, by another `DETERMINISTIC` setting, or by
  the old scripts (no `code_version.txt`), because `RESUME=1` would mix results.
  Use a new `NAME`, `RESUME=0`, or `FORCE=1`.
* `check` shows and checks the reference / obsdb folders that the cycle will actually
  read (`cycling.reference_dir` / `obsdb_root` from the config, else under `$DATA`), and
  exports them to the cycle. Before, it checked under `$DATA` while the cycle could read
  elsewhere.
* `plot` still uses `$DATA/plot_ls3dvar_cycling.py` and `$DATA/plot_ls3dvar.yaml`.

## Not yet in the package

Nothing the cycle needs. Still only in `mantari/`: everything else (the older
generations and the investigation scripts). See `docs/ls3dvar/CLEANUP_PLAN.md`.
