#!/bin/bash
# Usage: scripts/run_minifold12l_task.sh <index>
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: $0 <job-index>" >&2
  exit 1
fi

export CONFIG='{"jobs": [
  {"recycle": 1, "mode": "baseline", "override": null},
  {"recycle": 1, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 1, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 2, "mode": "baseline", "override": null},
  {"recycle": 2, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 2, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 3, "mode": "baseline", "override": null},
  {"recycle": 3, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 3, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 4, "mode": "baseline", "override": null},
  {"recycle": 4, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 4, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 5, "mode": "baseline", "override": null},
  {"recycle": 5, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 5, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 6, "mode": "baseline", "override": null},
  {"recycle": 6, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 6, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 7, "mode": "baseline", "override": null},
  {"recycle": 7, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 7, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 8, "mode": "baseline", "override": null},
  {"recycle": 8, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 8, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 9, "mode": "baseline", "override": null},
  {"recycle": 9, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 9, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"},
  {"recycle": 10, "mode": "baseline", "override": null},
  {"recycle": 10, "mode": "single", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides.pt"},
  {"recycle": 10, "mode": "multi", "override": "/var/tmp/famfold/test/gh5_21_cl80_template_overrides_top5.pt"}
]}'

export JOB_INDEX=$1
JOB=$(python - <<PY
import json
import os
jobs = json.loads(os.environ['CONFIG'])['jobs']
print(json.dumps(jobs[int(os.environ['JOB_INDEX'])]))
PY
)

RECYCLE=$(python -c "import json; print(json.loads('$JOB')['recycle'])")
MODE=$(python -c "import json; print(json.loads('$JOB')['mode'])")
OVERRIDE=$(python -c "import json; v=json.loads('$JOB')['override']; print('' if v is None else v)")

OUT_DIR_BASE="/var/tmp/famfold/test/minifold_12L_r${RECYCLE}_${MODE}_out"
FASTA=/var/tmp/famfold/test/gh5_21_subset.fasta
CACHE=/var/tmp/checkpoints
mkdir -p "$OUT_DIR_BASE"
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export TORCH_HOME=$CACHE
CMD=(uv run python predict.py "$FASTA" --out_dir "$OUT_DIR_BASE" --cache "$CACHE" --model_size 12L --token_per_batch 2048 --num_recycling "$RECYCLE")
if [ -n "$OVERRIDE" ]; then
  CMD+=(--template_overrides "$OVERRIDE")
fi

echo "Running recycle=${RECYCLE} mode=${MODE} override=${OVERRIDE}" >&2
"${CMD[@]}"
