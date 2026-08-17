# BCIAI — 耦合原生脑电情绪解码与策略社会

SEED-IV 原始波形上的直接迁移脑电情绪解码。主线成果是 DAME 架构（水循环 + 互助社会），
论文见 `paper/DAME_Paper_CN.md`（中文）与 `paper/DAME_Paper_EN.md`（英文）。

## 目录

- `eeg_v5_coupling_experiment.py` — DAME 主管线：耦合前端 / 水循环 / 互助社会 / 消融与双协议实验
- `eeg_v6_personality_society.py` — 性格异构社会实验（论文 VI-H 节）：三家族四维异构社会 vs 原社会
- `z_invariance_probe.py` — Z 场不变性直探针（论文 III-D 措辞的实证检验）
- `paper/` — 论文源文件（中英）与 PDF
- `results/` — 实验缓存（JSON，协议标记见各文件内 key）

## 运行

依赖 PyTorch (CUDA)、scikit-learn。数据为 SEED-IV 原始 EEG（62 通道），放置路径见
`eeg_v5_coupling_experiment.py` 中的数据加载段。

```bash
# DAME 主线 LOSO 消融（默认社会 = P6 异构社会，VI-H；缓存键带 _pers 后缀）
python eeg_v5_coupling_experiment.py --fast 8

# 复现论文表 3-5（原同构社会 MutualSocietyV3）
python eeg_v5_coupling_experiment.py --fast 8 --society mutual

# 性格异构社会（论文 VI-H）：跨会话 / 跨被试 两臂对照
python eeg_v6_personality_society.py --fast 8 --session-split --seeds 3
python eeg_v6_personality_society.py --fast 8 --seeds 3
```

## 纪律

全部实验预注册判读准则；负结果与正结果同等成文。证伪记录见论文附录 D / E 与 VI-H 节。
