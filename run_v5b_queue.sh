#!/bin/bash
# v5b 重跑队列 (2026-08-16 第二轮审查后)
# 顺序执行: 每个 job 完整跑完再进下一个, 便于中途断点续传 (done 缓存)
cd /c/Users/LENOVO/Desktop/BCIAI

PY=python
EP=15

echo "=== [1/4] CE-only 公平性消融 (LOSO 8s, seed 42) ==="
$PY eeg_v5_coupling_experiment.py --fast 8 --seeds 1 --epochs $EP \
    --models C5-CEOnly --tag v5b_ceonly || echo "JOB1 FAILED rc=$?"

echo "=== [2/4] NoMutual/Base 异常复核 (LOSO 8s, seed 123) ==="
$PY eeg_v5_coupling_experiment.py --fast 8 --seeds 1 --seed-start 1 --epochs $EP \
    --models C5-NoMutual,C5-Base --tag v5b_anom || echo "JOB2 FAILED rc=$?"

echo "=== [3/4] 跨会话四臂 × 3种子 (训练会话归一化修复后) ==="
$PY eeg_v5_coupling_experiment.py --fast 6 --seeds 3 --epochs $EP --session-split \
    --models C5-Base,C5-NoWaterNoPred,C5-NoPredNoMutual,C5-NoPred --tag v5b_xsess \
    || echo "JOB3 FAILED rc=$?"

echo "=== [4/4] 路由器复现 (跨会话, 场域路由器开启, × 3种子) ==="
$PY eeg_v5_coupling_experiment.py --fast 6 --seeds 3 --epochs $EP --session-split \
    --field-router --models C5-NoPred,C5-NoPredNoMutual --tag v5b_routerxsess \
    || echo "JOB4 FAILED rc=$?"

echo "=== QUEUE DONE ==="
