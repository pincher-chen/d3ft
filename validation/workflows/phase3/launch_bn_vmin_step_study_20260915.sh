#!/usr/bin/env bash
set -euo pipefail

root=${D3FT_ROOT}
base=$root/validation/results/phase3_20260908
seq=$base/sequences
python_bin=/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python
runner=$root/validation/workflows/phase3/run_gpu_sequence.py
waiter=$root/validation/workflows/phase3/wait_for_parameter_fd_pair.py
stamp=$(date +%Y%m%d_%H%M%S)

archive_failed_launch() {
    path=$1
    if [[ -e $path ]]; then
        mv "$path" "${path}.sandbox_failed_${stamp}"
    fi
}

launch_sequence() {
    plan=$1
    record=$2
    log=$3
    setsid "$python_bin" -u "$runner" "$plan" --record "$record" \
        > "$log" 2>&1 < /dev/null &
    launched_pid=$!
}

record_m02=$seq/p2_bn_vmin_a020_minus_4gpu233252
record_p02=$seq/p2_bn_vmin_a020_plus_4gpu233254g03
record_m08=$seq/p2_bn_vmin_a080_minus_4gpu233254g03_after_a002
record_p08=$seq/p2_bn_vmin_a080_plus_4gpu233254g47

for path in "$record_m02" "$record_p02" "$record_m08" "$record_p08"; do
    archive_failed_launch "$path"
done
for path in \
    "$seq/p2_bn_vmin_a020_minus_4gpu233252.controller.log" \
    "$seq/p2_bn_vmin_a020_plus_4gpu233254g03.controller.log" \
    "$seq/p2_bn_vmin_a080_minus_4gpu233254g03_after_a002.controller.log" \
    "$seq/p2_bn_vmin_a080_plus_4gpu233254g47.controller.log"; do
    archive_failed_launch "$path"
done

launch_sequence \
    "$base/p2_bn_vmin_a020_minus_4gpu233252_plan.json" "$record_m02" \
    "$seq/p2_bn_vmin_a020_minus_4gpu233252.controller.log"
pid_m02=$launched_pid
launch_sequence \
    "$base/p2_bn_vmin_a020_plus_4gpu233254g03_plan.json" "$record_p02" \
    "$seq/p2_bn_vmin_a020_plus_4gpu233254g03.controller.log"
pid_p02=$launched_pid
launch_sequence \
    "$base/p2_bn_vmin_a080_plus_4gpu233254g47_plan.json" "$record_p08" \
    "$seq/p2_bn_vmin_a080_plus_4gpu233254g47.controller.log"
pid_p08=$launched_pid

"$python_bin" - "$base/p2_bn_vmin_a080_minus_4gpu233254g03_after_a002_plan.template.json" \
    "$base/p2_bn_vmin_a080_minus_4gpu233254g03_after_a002_plan.json" \
    "$record_p02" "$pid_p02" <<'PY'
import json
import sys

source, destination, dependency, pid = sys.argv[1:]
plan = json.load(open(source))
plan["after_record"] = dependency
plan["after_pid"] = int(pid)
open(destination, "w").write(json.dumps(plan, indent=2) + "\n")
PY

launch_sequence \
    "$base/p2_bn_vmin_a080_minus_4gpu233254g03_after_a002_plan.json" "$record_m08" \
    "$seq/p2_bn_vmin_a080_minus_4gpu233254g03_after_a002.controller.log"
pid_m08=$launched_pid

record_c02=$seq/p2_bn_vmin_a020_certificate_waiter
record_c08=$seq/p2_bn_vmin_a080_certificate_waiter
archive_failed_launch "$record_c02"
archive_failed_launch "$record_c08"
archive_failed_launch "$seq/p2_bn_vmin_a020_certificate_waiter.controller.log"
archive_failed_launch "$seq/p2_bn_vmin_a080_certificate_waiter.controller.log"

setsid "$python_bin" -u "$waiter" \
    "$base/evaluate_BN_g60_c100_xc4_h0025_k10" \
    "$base/evaluate_BN_prod_k10_h0025_fd_vmin_a020_minus_4gpu233252" \
    "$base/evaluate_BN_prod_k10_h0025_fd_vmin_a020_plus_4gpu233254g03" \
    --minus-record "$record_m02" --plus-record "$record_p02" \
    --minus-pid "$pid_m02" --plus-pid "$pid_p02" \
    --out "$base/BN_production_k10_vmin_a020_fd_certificate.json" \
    --record "$record_c02" --rtol 0.02 --poll-seconds 30 \
    > "$seq/p2_bn_vmin_a020_certificate_waiter.controller.log" 2>&1 < /dev/null &
pid_c02=$!

setsid "$python_bin" -u "$waiter" \
    "$base/evaluate_BN_g60_c100_xc4_h0025_k10" \
    "$base/evaluate_BN_prod_k10_h0025_fd_vmin_a080_minus_4gpu233254g03_after_a002" \
    "$base/evaluate_BN_prod_k10_h0025_fd_vmin_a080_plus_4gpu233254g47" \
    --minus-record "$record_m08" --plus-record "$record_p08" \
    --minus-pid "$pid_m08" --plus-pid "$pid_p08" \
    --out "$base/BN_production_k10_vmin_a080_fd_certificate.json" \
    --record "$record_c08" --rtol 0.02 --poll-seconds 30 \
    > "$seq/p2_bn_vmin_a080_certificate_waiter.controller.log" 2>&1 < /dev/null &
pid_c08=$!

"$python_bin" - "$base/p2_bn_vmin_step_study_20260915.json" \
    "$pid_m02" "$pid_p02" "$pid_m08" "$pid_p08" "$pid_c02" "$pid_c08" <<'PY'
import datetime
import json
import sys

path = sys.argv[1]
state = json.load(open(path))
state.update(
    state="launched_persistent",
    launched_at=datetime.datetime.now().astimezone().isoformat(),
    controller_pids={
        "a020_minus": int(sys.argv[2]),
        "a020_plus": int(sys.argv[3]),
        "a080_minus_waiting": int(sys.argv[4]),
        "a080_plus": int(sys.argv[5]),
        "a020_certificate": int(sys.argv[6]),
        "a080_certificate": int(sys.argv[7]),
    },
)
open(path, "w").write(json.dumps(state, indent=2) + "\n")
PY

sleep 5
for record in "$record_m02" "$record_p02" "$record_m08" "$record_p08" "$record_c02" "$record_c08"; do
    echo "STATUS $record"
    cat "$record/status.json"
done
