#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22e_export_and_build_qat_arms.sh -- export the four QDQ ONNX arms and build
# their .rknn, in one detached chain.
#
#   export : conda env `uav`   (torch 1.13.1, onnx 1.18.0)
#   build  : conda env `rknn`  (rknn-toolkit2 2.3.2, torch 2.4.0+cpu)
#
# HARD CONSTRAINT on the build: rknn.build(do_quantization=False) and NO
# mean_values / std_values -- see 22c_build_qdq_rknn.py and
# 17_probe_qdq_import.py (QDQ_ADOPTED / QDQ_BUILD_FAILED).
#
# Run (server):
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/pipeline/npu/ptq/22e_export_and_build_qat_arms.sh \
#       > /mnt/data/siping/ptq_work/logs/qat_export_build.out 2>&1 < /dev/null &
set -uo pipefail

REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/ptq_work/qat/trial_0080
LOGDIR=/mnt/data/siping/ptq_work/logs
cd "$REPO"

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh

echo "############ EXPORT (env uav) ############"
conda activate uav
bash manu/pipeline/npu/ptq/22b_export_qat_arms.sh
RC_EXPORT=$?
echo "[export] rc=$RC_EXPORT"

echo "############ BUILD (env rknn) ############"
conda activate rknn
bash manu/pipeline/npu/ptq/22d_build_qat_arms.sh
RC_BUILD=$?

echo ""
echo "############ SUMMARY ############"
echo "export rc=$RC_EXPORT   build rc=$RC_BUILD"
md5sum "$W"/onnx/*.onnx "$W"/rknn/*.rknn 2>&1
echo "[ALL DONE]"
