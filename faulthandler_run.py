"""包装器: 禁用 cuDNN (RTX5070 Blackwell + torch dev nightly 规避)
看门狗已移至 eeg_v5_coupling_experiment.py 训练循环内 (每 epoch 重 arm,
600s 无进度才 dump+exit)。此处不再放一次性 dump_traceback_later —
教训 2026-08-13: 一次性看门狗在进程第 600s 无条件杀进程, 误伤健康运行
(基线连续"伪崩溃": 每段日志都在 600s 整点 Timeout, 且重启后无 nvlddmkm 事件)。
"""
import faulthandler, sys, os

import torch
torch.backends.cudnn.enabled = False
print(f"[workaround] cudnn.enabled={torch.backends.cudnn.enabled}", flush=True)

import importlib.util
spec = importlib.util.spec_from_file_location("v5", "eeg_v5_coupling_experiment.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if __name__ == "__main__":
    m.main()
