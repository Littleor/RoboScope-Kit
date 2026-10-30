#!/usr/bin/env bash
set -Eeuo pipefail

SOFT=0
if [[ "${1:-}" == "--soft" ]]; then
  SOFT=1
fi

# Stop legacy and autotune benchmark processes from a previous run.
pkill -f "/root/starVLA/deployment/model_server/server_policy.py" 2>/dev/null || true
pkill -f "/root/starVLA/examples/LIBERO/eval_files/eval_libero.py" 2>/dev/null || true
pkill -f "benchmarks/libero_official/starvla_runtime/libero_server.py" 2>/dev/null || true
pkill -f "benchmarks/libero_official/starvla_runtime/libero_standard_worker.py" 2>/dev/null || true
pkill -f "benchmarks/libero_official/starvla_runtime/libero_client.py" 2>/dev/null || true
if [[ "$SOFT" != "1" ]]; then
  pkill -f "/root/roboscope_libero_runner.py" 2>/dev/null || true
fi
sleep 1
pkill -9 -f "/root/starVLA/deployment/model_server/server_policy.py" 2>/dev/null || true
pkill -9 -f "/root/starVLA/examples/LIBERO/eval_files/eval_libero.py" 2>/dev/null || true
pkill -9 -f "benchmarks/libero_official/starvla_runtime/libero_server.py" 2>/dev/null || true
pkill -9 -f "benchmarks/libero_official/starvla_runtime/libero_standard_worker.py" 2>/dev/null || true
pkill -9 -f "benchmarks/libero_official/starvla_runtime/libero_client.py" 2>/dev/null || true

rm -rf /root/benchmark
rm -rf /root/testvideo /root/testresult
mkdir -p /root/testvideo /root/testresult
cat > /root/testlog.txt <<LOG
用户模型权重下载进度 0%
测试套件下载进度 0%
LIBERO none 0 0 0 idle
LOG

# Remove runtime checkpoint links or downloads created for evaluation.
rm -rf /root/starVLA/playground/Checkpoints/*
rm -f /root/testresult/policy_server.log /root/testresult/evaluator.log

exit 0
