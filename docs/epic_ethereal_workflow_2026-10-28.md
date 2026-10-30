# 空灵史诗 / 战斗 BGM 工作流（2026-10-28）

> 背景：2026-10-26 交付的 8 首中文歌被用户判为"糖水歌"。用户新 brief：
> 要有**史诗感**和**空灵感**，不要糖水歌；参考《新·与天齐》《自由的你》《诀爱》
> 与热门战斗 BGM。本文记录诊断、新的写法/管线与验收口径。

---

## 1. 为什么上一批是"糖水"：实测

用 `tools/generation/style_qa.py` 把 8 首交付曲和 `human_refs/refs/` 里
12 首本地史诗参考（TSFH《Victory》、HOYO-MiX《Wildfire》《Flares of the
Blazing Sun》、Nightwish、Sabaton、Skillet、Ruelle 等）放在同一把尺子上：

| 指标 | 10-26 的 8 首 | 12 首史诗参考 | 结论 |
|---|---|---|---|
| 调式 | 6/8 首小调（并不"甜"在调式） | 多为小调，也有大调 | 不是主因 |
| 立体声宽度 | **0.25–0.56** | **0.38–0.84**（中位 ~0.57） | 明显偏窄 |
| 2–6 kHz 相对 0.5–2 kHz | **-11 ~ -17 dB** | **-7 ~ -13 dB** | 中高频"咬合感"不足，发闷 |
| 30–120 Hz | +27 ~ +30 dB | +19 ~ +38 dB（多数 ≥29） | 低频量够，但没托起来 |
| 动态范围（1 s RMS P95–P10） | 10.7–19.4 dB | 1.3–29.2 dB | 不是主因 |
| `style_qa` epic 分 | **20.8–56.7，7/8 判 soft** | — | 量化复现了听感 |

根因不在"用了大调"，而在三件事：

1. **标签在要糖水。** `build_tags()` 每一档人声都以
   `sweet clear young female vocal` / `clear young female vocal` 开头，
   尾巴固定跟 `polished modern pop`；叙事档还叠加 `warm piano` /
   `shimmering guitars` / `tender intimate delivery`。
2. **作词系统提示写的是清甜流行。** 旧 `SYSTEM_PROMPT` 第一句就是
   "你为一位清甜女声写歌……具体、口语"，epic 模式只是加了附加条款，
   底子仍是近景叙事；02/03 两首直接写成了便利店/雨伞/电梯的日常小品。
3. **编曲没有"史诗"的物理特征。** 弦乐/合唱/战鼓写进了标签，但混音是
   窄而暗的流行混音；2–6 kHz 的存在感和宽度都低于参考曲。

---

## 2. 三条主线（本轮只做校准批，一首一个方向）

| # | 类型 | 参考 | 调/BPM | 人声 | 编曲关键词 |
|---|---|---|---|---|---|
| 01 | 空灵史诗 `epic_ethereal` | 《诀爱》 | A minor / 74 | 飘逸头声、气声、位置远、长混响 | 弦乐、低音大提琴 drone、钢琴、天使合唱、定音鼓、大厅混响、宽声场 |
| 02 | 战斗史诗 `battle_anthem` | 《新·与天齐》 | D minor / 152 | 御姐胸混声、咬字清楚、副歌齐唱感 | 太鼓、二胡、笛子、古筝、弦乐、铜管、合唱 |
| 03 | 哲思史诗 `epic` | 《自由的你》 | A minor / 100 | 御姐低中音、直给、副歌放开 | 弦乐、钢琴、合唱、戏剧性推进、末副歌满编 |
| 04 | 战斗 BGM（器乐） | 热门战斗 BGM / TSFH | D minor / 148 | 无人声 | TSFH LoRA：巨大交响、太鼓、铜管、天使/黑暗合唱 |

作词模式（`zh_lyrics_gen.py` / `zh_lyric_qa.py`）：

- 新增 `epic_ethereal`（20–28 句，主歌 3–5 句、副歌 3–5 句，词少、留白）；
- 新增 `battle_anthem`（22–30 句，口号 hook 4+3 切分，副歌只有第一句是 hook）；
- 两者都走新的 `EPIC_SYSTEM_PROMPT`：史诗尺度场景、一个中心比喻、
  桥段反转、具体物件托住抽象命题；**禁止**便利店/末班车/雨伞/电梯/外卖等近景。

标签（`build_tags()`）：

- 史诗档不再出现 `sweet` / `young` / `tender` / `intimate` / `polished modern pop`；
- 新增 `no pop production` / `no acoustic guitar` / `no sweet soft tone` 反向约束；
- 人声描述按类型分开：
  - 空灵：`ethereal female vocal, floating high head voice, airy and breathy, distant and vast`
  - 战斗：`powerful female vocal, heroic chest-mix belt, clear diction`
  - 哲思：`powerful female vocal, clear chest-mix belt, emotional and direct`
- 生成器的 `validate_brief` 会拦截模型自己写出的 `no vocals` / `instrumental` /
  糖水词，避免自我矛盾。

---

## 3. LoRA：本地已有、此前未接

`models/yue2/loras/` 里躺着两套与本次目标完全对口的 LoRA：

- `tsfh-steps-from-hell/adapter-ar-195 + adapter-nar-194`
  （Two Steps From Hell 史诗管弦乐；AR 改计划/结构，NAR 改音色；
  推荐强度 AR 0.5 / NAR 0.375）；
- `instrumental-cot`（AR 分支，器乐 + ABC 结构计划，rank 64）。

YuE2 推理包（0.1.3/0.1.5）没有 LoRA 钩子，但适配器张量名与模型模块名完全对齐
（`model.layers.{i}.self_attn.q_proj.lora_A/B`、`nar_self_attn` / `nar_mlp`、
`vae2llm` / `llm2vae`）。新增 `tools/generation/yue2_lora.py` 在加载后把
`W += scale * alpha/rank * (B @ A)` 折进基座权重，不修改 vendor 包。

实测：AR 196 个 Linear、NAR 198 个（含 `vae2llm`/`llm2vae`）全部命中，
无未解析张量；单测 4 项通过。

---

## 4. 管线

```
选题卡（seeds.json）
  -> zh_lyrics_gen.py --seeds-file（DeepSeek + 本地门控）
  -> zh_lyric_qa.py（韵/声调/段落/hook/抽象词/具体物件）
  -> YuE2 生成 1 候选/方向（原生 48 kHz，无 SVC/EQ/压缩/限幅）
       plan_shape: hook chorus + thin verse:2 + final-lift 2
       器乐线：TSFH LoRA（AR 0.5 / NAR 0.375），[instrumental] 歌词
  -> arc_qa（欲扬先抑）+ diction_qa（中文覆盖）
  -> style_qa（调式/宽度/2–6k 存在感/sub/centroid -> epic 0–100）
  -> 人耳听审（最终标准）
```

`run_epic_20261028.py`：一首一进程（Windows CUDA 分配器碎片化），
逐曲写日志与 `logs/epic_20261028_status.json`，产出后自动跑 `style_qa`。

---

## 5. 验收口径（本轮新增）

`tools/generation/style_qa.py` 的 epic 分（0–100）：

| 项 | 权重 | 标尺 |
|---|---:|---|
| 立体声宽度 | 35% | 0.32 -> 0；0.68 -> 1 |
| 2–6 kHz 相对 0.5–2 kHz | 35% | -17 dB -> 0；-8.5 dB -> 1 |
| sub 30–120 Hz | 15% | 24 dB -> 0；34 dB -> 1 |
| 频谱质心 | 15% | 1500 Hz -> 0；2400 Hz -> 1 |

判定：≥60 `wide`，35–60 `typical`，<35 `narrow`。
这是**红灯**，不是听审替代品：10-14/15/16 已被接受的英文史诗批在这把尺上
也只有 29–60，而 10-26 被拒的中文批是 20–57，两者重叠。所以它只能标记
"特别窄/暗"的混音，不能判断"史诗感"；真正的判据仍是 `arc_qa`（欲扬先抑）、
`diction_qa`（咬字）和人耳。

---

## 6. 待确认

1. 校准批 4 首出片后，用户先选方向；选定后再按方向扩到 6–8 首。
2. 空灵线的"气声/远"与 YuE2 的可懂度可能冲突；若咬字糊，改
   `airy and breathy` -> `clear ethereal tone with reverb`，其余不动。
3. TSFH LoRA 只用于器乐线；若想用在带人声曲，先用 AR 0.3 短样验证
   人声是否还在（NAR LoRA 会把人声推向器乐音色，默认不叠加）。
