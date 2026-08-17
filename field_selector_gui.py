#!/usr/bin/env python3
"""场域选择器弹窗 — 手动选协议 → 自动分配架构 → 运行 → 弹窗输出结果与预测
用法: python field_selector_gui.py

架构自动分配规则 (来自已裁决实验证据):
  跨被试 (LOSO)          → C5-NoPredNoMutual (纯水循环通路 — 社会跨被试为负, 已证)
  跨会话 (train s1+s2→test s3) → C5-NoPred (水循环+互助社会 — 社会跨会话转正 +0.86, 已证)

弹窗输出: 训练进度 / 各臂 acc / 机制贡献 Δ / 每类准确率 / 混淆矩阵 / 预测分布
落盘: results/gui_runs.jsonl (独立日志, 不污染 v5_iterations.jsonl)
2026-08-14
"""
import sys
import os
import json
import queue
import threading
import time
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import eeg_v5_coupling_experiment as E

CLASS_NAMES = ["neutral", "sad", "fear", "happy"]

PROTOCOLS = {
    "loso": {
        "label": "跨被试 (LOSO)",
        "arch": "C5-NoPredNoMutual",      # 纯水循环
        "control": "C5-NoPred",           # 水+社会
        "arch_desc": "水循环通路 (社会在跨被试为负 -1.86, 已证关闭)",
    },
    "session": {
        "label": "跨会话 (train s1+s2 → test s3)",
        "arch": "C5-NoPred",              # 水+社会
        "control": "C5-NoPredNoMutual",
        "arch_desc": "水循环 + 互助社会通路 (社会跨会话转正 +0.86, 已证开启)",
    },
}

_DATA_CACHE = {}   # n_subjects -> (X, y, subj, pair_idx, wid)


class _QWriter:
    """把实验进程的 print 流导入弹窗日志."""
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s.strip():
            self.q.put(s)

    def flush(self):
        pass


def load_cached(n_subjects, q):
    if n_subjects not in _DATA_CACHE:
        t0 = time.time()
        X, y, subj, pair_idx, wid = E.load_raw_with_pairs(n_subjects=n_subjects)
        _DATA_CACHE[n_subjects] = (X, y, subj, pair_idx, wid)
        q.put(f"[load] 数据准备 {time.time() - t0:.0f}s (缓存)")
    return _DATA_CACHE[n_subjects]


def run_worker(proto_key, n_subjects, seeds, epochs, q, out):
    """后台线程: 数据 → 双协议按架构路由 → 运行双臂 → 收集预测."""
    old = sys.stdout
    sys.stdout = _QWriter(q)
    spec = PROTOCOLS[proto_key]
    try:
        print(f"[场域选择器] 协议={spec['label']}")
        print(f"[路由] 自动分配架构 = {spec['arch']} ({spec['arch_desc']})")
        print(f"[路由] 对照臂 = {spec['control']}")
        X, y, subj, pair_idx, wid = load_cached(n_subjects, q)

        arch_spec = E.MODEL_SPECS[spec["arch"]]
        ctrl_spec = E.MODEL_SPECS[spec["control"]]
        collector = []     # [(key, acc, pred, y_true)]

        if proto_key == "loso":
            res_arch = E.loso_v5(arch_spec["fn"], X, y, subj, pair_idx, seeds,
                                 epochs=epochs, kind=arch_spec["kind"],
                                 tag="gui_loso_arch", collector=collector)
            res_ctrl = E.loso_v5(ctrl_spec["fn"], X, y, subj, pair_idx, seeds,
                                 epochs=epochs, kind=ctrl_spec["kind"],
                                 tag="gui_loso_ctrl", collector=collector)
        else:
            res_arch = E.session_run_v5(arch_spec["fn"], X, y, subj, wid, pair_idx,
                                        seeds, epochs=epochs, kind=arch_spec["kind"],
                                        tag="gui_sess_arch", collector=collector)
            res_ctrl = E.session_run_v5(ctrl_spec["fn"], X, y, subj, wid, pair_idx,
                                        seeds, epochs=epochs, kind=ctrl_spec["kind"],
                                        tag="gui_sess_ctrl", collector=collector)
        out.put(("done", res_arch, res_ctrl, collector, spec))
    except Exception as ex:                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        out.put(("error", repr(ex)))
    finally:
        sys.stdout = old


def summarize(res_arch, res_ctrl, collector, spec):
    """acc / 机制贡献 / 每类准确率 / 混淆矩阵 / 预测分布 → 文本."""
    arch_acc = float(np.mean(list(res_arch.values())))
    ctrl_acc = float(np.mean(list(res_ctrl.values())))
    delta = arch_acc - ctrl_acc
    sign = "社会有用 (+)" if delta > 0 else ("社会拖累 (-)" if delta < 0 else "中性")

    arch_pred, arch_true = [], []
    for key, acc, pred, yt in collector:
        if "_arch_" in key:
            arch_pred.append(pred)
            arch_true.append(yt)
    p = torch.cat(arch_pred).numpy()
    t = torch.cat(arch_true).numpy()

    n = E.N_CLASSES
    conf = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            conf[i, j] = int(((t == i) & (p == j)).sum())
    per_class = [conf[i, i] / max(conf[i].sum(), 1) for i in range(n)]
    pred_dist = [int((p == j).sum()) for j in range(n)]

    lines = []
    lines.append("=" * 58)
    lines.append("运行完成 — 结果与预测")
    lines.append("=" * 58)
    lines.append(f"协议: {spec['label']}")
    lines.append(f"自动分配架构: {spec['arch']}  ({spec['arch_desc']})")
    lines.append(f"对照臂:       {spec['control']}")
    lines.append("-" * 58)
    lines.append(f"[{spec['arch']}] 平均 acc = {arch_acc * 100:.2f}%  "
                 f"(逐种子: {', '.join(f'{v * 100:.2f}' for v in res_arch.values())})")
    lines.append(f"[{spec['control']}] 平均 acc = {ctrl_acc * 100:.2f}%  "
                 f"(逐种子: {', '.join(f'{v * 100:.2f}' for v in res_ctrl.values())})")
    lines.append(f"机制贡献 Δ = {delta * 100:+.2f} 点 → {sign}")
    lines.append("-" * 58)
    lines.append(f"每类准确率 (recall, {spec['arch']} 全折累计 {len(p)} 窗):")
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:8s}: {per_class[i] * 100:6.2f}%  (真值 {conf[i].sum():5d} 窗)")
    lines.append(f"预测分布: " + ", ".join(f"{CLASS_NAMES[j]}={pred_dist[j]}" for j in range(n)))
    lines.append("混淆矩阵 (行=真值, 列=预测):")
    head = "        " + "".join(f"{name[:4]:>8s}" for name in CLASS_NAMES)
    lines.append(head)
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:5s} " + "".join(f"{conf[i, j]:8d}" for j in range(n)))
    lines.append("=" * 58)
    return "\n".join(lines), {
        "protocol": spec["label"], "arch": spec["arch"], "control": spec["control"],
        "arch_acc": arch_acc, "ctrl_acc": ctrl_acc, "delta": delta,
        "per_class": per_class, "confmat": conf.tolist(), "pred_dist": pred_dist,
    }


def make_gui():
    import tkinter as tk
    from tkinter import scrolledtext, ttk

    root = tk.Tk()
    root.title("DAME 场域选择器 — 手动选协议 → 自动分配架构")
    root.geometry("860x620")

    proto_var = tk.StringVar(value="session")

    def arch_line():
        sp = PROTOCOLS[proto_var.get()]
        return f"自动分配架构: {sp['arch']} — {sp['arch_desc']}"

    tk.Label(root, text="① 选择协议 (场域)", font=("Microsoft YaHei", 11, "bold")
             ).pack(anchor="w", padx=12, pady=(10, 2))
    for key, sp in PROTOCOLS.items():
        tk.Radiobutton(root, text=sp["label"], variable=proto_var, value=key,
                       font=("Microsoft YaHei", 10)).pack(anchor="w", padx=28)
    arch_label = tk.Label(root, text=arch_line(), font=("Microsoft YaHei", 9),
                          fg="#2e8b57")
    arch_label.pack(anchor="w", padx=28, pady=(2, 8))

    def on_proto(*_):
        arch_label.config(text=arch_line())

    proto_var.trace_add("write", on_proto)

    tk.Label(root, text="② 参数", font=("Microsoft YaHei", 11, "bold")
             ).pack(anchor="w", padx=12, pady=(4, 2))
    pf = tk.Frame(root)
    pf.pack(anchor="w", padx=12)
    vars_ = {}
    for i, (name, default, span) in enumerate([("被试数", "6", (4, 15)),
                                               ("种子(逗号分隔)", "42", None),
                                               ("轮数", "15", (2, 30))]):
        tk.Label(pf, text=name, font=("Microsoft YaHei", 9)).grid(row=0, column=i * 2, padx=2)
        e = tk.Entry(pf, width=8, font=("Consolas", 10))
        e.insert(0, default)
        e.grid(row=0, column=i * 2 + 1, padx=(2, 10))
        vars_[name] = e

    status = tk.StringVar(value="就绪")
    run_btn = tk.Button(root, text="③ 开始运行", font=("Microsoft YaHei", 10, "bold"),
                        bg="#2e8b57", fg="white", height=1, width=14)
    run_btn.pack(anchor="w", padx=12, pady=6)

    out_box = scrolledtext.ScrolledText(root, font=("Consolas", 9), height=26)
    out_box.pack(fill="both", expand=True, padx=12, pady=(2, 4))

    tk.Label(root, textvariable=status, font=("Microsoft YaHei", 9), fg="#555"
             ).pack(anchor="w", padx=12, pady=(0, 8))

    q = queue.Queue()

    def start():
        try:
            n_subj = int(vars_["被试数"].get())
            seeds = [int(s.strip()) for s in vars_["种子(逗号分隔)"].get().split(",") if s.strip()]
            epochs = int(vars_["轮数"].get())
        except ValueError:
            out_box.insert("end", "[参数错误] 被试数/种子/轮数必须是整数\n")
            return
        est = "约 25-35 分钟" if proto_var.get() == "loso" else "约 6-10 分钟"
        run_btn.config(state="disabled")
        status.set(f"运行中 ({est})...")
        out_box.delete("1.0", "end")
        threading.Thread(target=run_worker, daemon=True,
                         args=(proto_var.get(), n_subj, seeds, epochs, q, q)
                         ).start()

    def poll():
        try:
            while True:
                msg = q.get_nowait()
                if isinstance(msg, tuple) and msg and msg[0] == "done":
                    _, res_arch, res_ctrl, collector, spec = msg
                    text, record = summarize(res_arch, res_ctrl, collector, spec)
                    out_box.insert("end", text + "\n")
                    with open("results/gui_runs.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            **record, "n_subjects": int(vars_["被试数"].get()),
                                            "seeds": vars_["种子(逗号分隔)"].get(),
                                            "epochs": int(vars_["轮数"].get())},
                                           ensure_ascii=False) + "\n")
                    run_btn.config(state="normal")
                    status.set("完成 — 结果已写入 results/gui_runs.jsonl")
                elif isinstance(msg, tuple) and msg and msg[0] == "error":
                    out_box.insert("end", f"[错误] {msg[1]}\n")
                    run_btn.config(state="normal")
                    status.set("出错")
                else:
                    out_box.insert("end", str(msg) + "\n")
                out_box.see("end")
        except queue.Empty:
            pass
        root.after(150, poll)

    run_btn.config(command=start)
    out_box.insert("end", "使用: ① 选协议 → ② 设参数 → ③ 开始运行\n"
                   "实验打印与最终结果/预测都输出在本窗口。\n\n")
    root.after(150, poll)
    return root


if __name__ == "__main__":
    make_gui().mainloop()
