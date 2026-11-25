# hrrrcast-ls3dvar container

HRRRCast-live fork + LS-3DVar (blocks control vector, no PCA) + realtime obs (IEM ASOS, MRMS),
for any HRRR subdomain in `domains/<name>.env` (`socal`, `fl_panhandle`).

## Host folders (mounted, never in the image)
| mount | host | contents |
|---|---|---|
| `/models` (ro) | `$MODELS_DIR` | HRRRCast `net-diffusion/`: `model.keras`, `normalize-stats.nc` |
| `/data` | `$DATA_DIR` | `runs/<domain>/` reference, `mrms/`, `ioda/`, `obsdb/`, `experiments/`, `wxbox/datasets/iem-asos` mirror |
| `/cache/nv` | `$NV_CACHE` | CUDA PTX-JIT cache (Blackwell): first run compiles ~30 min, then cached |

Optional Mantari feeds: put/mount IODA files at `$DATA_DIR/ioda/tempest`, `$DATA_DIR/ioda/grenade`; used when present.

## Run
```bash
export MODELS_DIR=~/models/net-diffusion DATA_DIR=~/hrrrcast-data
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm ls3dvar check
docker compose -f docker/compose.yaml run --rm ls3dvar cycle 2026-11-24T12 2026-11-24T14
DOMAIN=fl_panhandle docker compose -f docker/compose.yaml run --rm ls3dvar cycle 2026-11-24T12 2026-11-24T14
```
Steps can run alone: `stage | mrms | asos | obsdb START [END]`; `SKIP_PREP=1 ... cycle` reuses staged inputs.
`cycle` stages START-1h .. END+max(LEADS) (first background = F01 of START-1h; verification at H+lead).

Settings (env): `DOMAIN`, `LEADS` (`[1, 2, 3]`), `HOLDOUT` (`null` = assimilate all; `0/5` = withhold 1/5 for scoring),
`ASOS_SKIP_SYNC=1`, `SKIP_PREP=1`. Output: `$DATA_DIR/experiments/<domain>/<domain>_<START>/` (`verification*.csv`).

## New domain
`domains/<name>.env`: `SUB_BBOX=N,W,S,E`, `SUB_HALO=40`, `DOMAIN_NAME`, `ASOS_NETWORKS="XX_ASOS ..."`.
`python src/crop_domain.py --bbox N,W,S,E --halo 40 --dry-run` gives the crop (height%8==3, width%8==7).
