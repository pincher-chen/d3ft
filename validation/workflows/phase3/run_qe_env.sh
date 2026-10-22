#!/usr/bin/env bash
# Intel-linked QE with Python/NumPy, without activating a competing MPI runtime.
set -eo pipefail
source /APP/u22/ai_x86/intel/oneapi2023.2/setvars.sh --force >/dev/null 2>&1
export QE_EXE=/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_box/qe-7.5/bin_cpu_intel/bin/pw.x
export D3FT_REFERENCE_ROOT=/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/Performance01/chenp/d3ft
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${QE_OMP_THREADS:-8}"
exec /HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_install/miniconda3/envs/jrystal/bin/python "$@"
