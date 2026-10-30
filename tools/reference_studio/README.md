# 参考编曲工作台

把参考歌拆为可修改的段落表，再做有对照的局部实验。此工具不把声学统计值当“热门程度”“钩子相似度”或歌曲质量分。

第一轮选择16号第4首 Iron Rain 的首主歌到首副歌，共21小节，符号计划约58.605秒。保留当前旋律、歌词、和弦、速度与唱法提示，先验证 YuE2 能否听从 `V: Ins`，再决定是否扩整曲。

## 四个阶段

1. `init-annotations`：用4秒能量、频带、瞬态特征的局部变化提出声学边界；最小间隔默认16秒。输出 `section_01` 等未审标签，不能自动当主歌/副歌。
2. `blueprint`：读取参考音频和人工可修正的段落标注，逐段输出节奏、能量、频带活动；如提供时间对齐的乐器分轨，则测量各轨活动。没有分轨时乐器身份保持未知。
3. `prepare` / `render`：生成固定条件的 baseline / sparse / full / arranged 四臂。每臂用新子进程渲染、单独保存日志和断点。
4. `report`：汇总真实完成音频、生成同响度试听与人工评分模板。

## 运行环境

在项目根目录执行命令。CPU分析需要 NumPy、SoundFile、ffmpeg/ffprobe，本机已有：

```powershell
$CpuPy = 'tools/vocal_repair/.venv/Scripts/python.exe'
$YuePy = 'vendor/YuE2/.venv/Scripts/python.exe'
```

准备实验不需要GPU。渲染使用本地 YuE2 权重；不接商用云，不下载参考歌或模型。

## 参考段落与蓝图

先生成声学边界初稿：

```powershell
& $CpuPy -m tools.reference_studio init-annotations `
  --reference human_refs/refs/hoyo-mix_wildfire.m4a `
  --out outputs/reference_studio/wildfire.annotations.json
```

打开JSON，听歌调整时间、命名段落，并写乐器进出。缓存转录或声学推断得到的标签必须标为 `provisional`，听过后才能标为 `reviewed`。例：

```json
{
  "schema": 1,
  "status": "provisional_not_listened",
  "tempo_bpm": null,
  "first_beat_s": null,
  "beats_per_bar": 4,
  "sections": [
    {
      "name": "verse",
      "start_s": 0,
      "end_s": 24,
      "status": "provisional",
      "roles": {},
      "target": {
        "plan_section": "verse",
        "ins_pattern": "sparse",
        "proposal_status": "hypothesis_requires_listening"
      }
    },
    {
      "name": "chorus",
      "start_s": 24,
      "end_s": 48,
      "status": "provisional",
      "roles": {},
      "target": {
        "plan_section": "chorus",
        "ins_pattern": "full",
        "proposal_status": "hypothesis_requires_listening"
      }
    }
  ]
}
```

上面只是格式示例，时间并非对 Wildfire 的实测标注。区间可以有空档，不能重叠、倒序或超过文件时长。标注 `tempo_bpm` 和 `first_beat_s` 后才能得到基于节拍相位的瞬态分布；该分布仍不是经过验证的律动好坏分。

```powershell
& $CpuPy -m tools.reference_studio blueprint `
  --reference human_refs/refs/hoyo-mix_wildfire.m4a `
  --annotations outputs/reference_studio/wildfire.annotations.json `
  --out outputs/reference_studio/wildfire.blueprint.json
```

可选 `--stems stems.json`，JSON为 `{"drums":"D:/.../drums.wav","bass":"D:/.../bass.wav"}`。分轨必须与参考曲从0秒对齐、时长一致，相对路径按标注文件所在目录解析。工具不会把低频活动自动命名为贝斯，也不会把高频瞬态命名为镲片。人工 `roles` 原样保留，与实测分轨指标分开。

蓝图每段包含400毫秒RMS变化、峰值/RMS差、30–160Hz / 160–2000Hz / 2–10kHz能量与相对活动、混音瞬态密度、未核实节拍估计。`target.ins_pattern` 为待试听假设，可手改；它不是提取出的配器真值。

### 将参考区间映射到目标计划

声学区间仍叫 `section_XX` 时，默认不会假定其是verse或chorus。先提出一个可编辑映射，输出到新文件：

```powershell
& $CpuPy -m tools.reference_studio propose-mapping `
  --blueprint outputs/reference_studio/wildfire.blueprint.json `
  --out outputs/reference_studio/wildfire.mapped.blueprint.json
```

它选择能量和混音瞬态变化最大的相邻两个参考区间，把较低活动区间提议为sparse、较高活动区间提议为full；按时间顺序填入目标计划的 `verse` / `chorus` 槽位。最大变化若是收束，则可能提出verse full、chorus sparse，必须听过并调整方向。这些槽位不是对参考区间身份的判断；原 `section_XX`、测量和未审状态全部保留，标记为 `provisional_source_role_unknown`。

可用 `--source-sections section_03 section_04` 指定参考区间，用 `--target-sections verse chorus` 指定两个不同目标槽位。已有显式映射会拒绝重写，除非提供 `--replace-mapping`；输入蓝图不会被覆盖。没有可测对比时不编造映射。

人工可直接编辑输出中的 `target.plan_section` 与 `target.ins_pattern`。然后把 `prepare --blueprint` 指向此新文件。manifest和CLI会报告实际匹配的目标段落、未匹配段落、未使用的源区间及来源：

- `source_blueprint`：所有目标段落确实使用参考映射。
- `mixed_source_blueprint_and_generic_control`：只有部分目标段落使用参考映射。
- `generic_control`：没有匹配，明确提示此为通用Ins可控性实验，不能称参考驱动。

baseline/sparse/full仍是固定控制臂，只有arranged应用参考映射。匹配成立也只说明输入被应用，不代表听感或配器达到目标。

## 准备58秒完整主副歌对照

```powershell
$Experiment = 'outputs/reference_studio/iron_rain_wildfire_e2'
& $CpuPy -m tools.reference_studio prepare `
  --plan outputs/2026-10-16_epic_hook/candidates/04_iron_rain_20261016-150430/04_iron_rain_c2_artifacts/score.abc `
  --lyrics inputs/2026-10-15_epic_en/04_iron_rain_lyrics.txt `
  --style inputs/2026-10-15_epic_en/04_iron_rain_tags.txt `
  --blueprint outputs/reference_studio/wildfire.blueprint.json `
  --out-dir $Experiment
```

默认从第一个 `% verse` 开始保留两个连续段落。也可用 `--sections verse:1 chorus:1`，只允许相邻段落；改选第二次主副歌时会选择对应的第二组歌词。ABC把预副歌放入 verse 时，自动保留原歌词里的 `[Pre-Chorus]`。无法一一匹配则要求显式 `--excerpt-lyrics`，避免把错误歌词悄悄套入计划。

默认要求完整片段在30–60秒；更长实验需显式 `--allow-long`。再次执行相同输入可复用；输入不同必须换输出目录。准备阶段不覆盖已有实验的音频。

四臂设计：

| 臂 | Ins操作 | 用途 |
| --- | --- | --- |
| baseline | 原计划片段，保留Ins | 同计划基线 |
| sparse | 每小节半小节根音、半小节休止 | 极稀疏压力测试 |
| full | 根音/三音/五音的八次等分分解 | 高密度压力测试 |
| arranged | 按蓝图映射，默认主歌稀疏、副歌展开 | 段落对比假设 |

每次改写检查所有Ins小节时值，并检查Vocal和header哈希不变。Vocal里的和弦标记、旋律、停顿、连音保持原样。源Ins若多小节休止写作 `Z4` 会按整小节解析。源Ins与Vocal小节数不一致会在清单报告，编辑臂以Vocal时长为准；若多余Ins小节包含和弦则拒绝处理。未知ABC语法、和弦性质或每小节多次换和弦会拒绝，要求人工计划。

`Ins`仍是单个旋律声部，不是鼓组、吉他、贝斯、弦乐的独立轨道。full中的“展开”指符号密度，并不保证模型渲染成全乐队。无法保证具体乐器进出时，应转MIDI/DAW做分轨编曲。

### 温和support对照

首轮Ins实验发现：相同21小节/58.605秒计划，baseline实际58.639秒，繁忙arranged实际81.559秒，偏长约39.17%。外部ABC在本地YuE2中作为文本条件进入AR；实际结束由semantic采样决定，并没有按ABC小节强制时钟。数学时值合法不代表模型一定遵守，不能把这个首轮样本当成已验证的参考编曲升级。

下一组可以在新目录准备 `--arranged-pattern support`：

```powershell
& $CpuPy -m tools.reference_studio prepare `
  --plan outputs/2026-10-16_epic_hook/candidates/04_iron_rain_20261016-150430/04_iron_rain_c2_artifacts/score.abc `
  --lyrics inputs/2026-10-15_epic_en/04_iron_rain_lyrics.txt `
  --style inputs/workflows/your_short_style.txt `
  --blueprint outputs/reference_studio/wildfire.mapped.blueprint.json `
  --arranged-pattern support `
  --out-dir outputs/reference_studio/iron_rain_support
```

support仅改变arranged臂：sparse每小节一个根音加半小节休止，full每小节根音、五度各占1/4小节，再休止1/2小节。它只尝试伴奏密度对比，避免八次分解演奏形成独立繁忙旋律。baseline和sparse/full压力测试臂保持原定义。实际时长、歌词和听感仍需重新验证。若同时换短style，旧baseline不再输入等价，必须生成新的baseline，不能复用旧caption的结果。

### melody模式与歌词条件强度

`prepare`可加 `--cot melody --cfg-scale 1.5` 或 `--cfg-scale 2`。默认仍是 `--cot full`，不覆盖YuE2原生cfg默认。旧manifest缺少配置时也保持该行为，已有排队实验不用改JSON。

melody模式先按原和弦构建Ins，再仅删除Vocal/Ins音乐行中的和弦引号符号；header中的声部name引号、音符、休止、连线和时值保留，转换记录在 `cot_conversion`。配置进入manifest、输入identity、worker实际请求和结果。不同cot/cfg不能复用相同音频；源原生request/config也必须与配置一致。

本地YuE2的cfg=1时不运行negative branch；cfg>1放大带style/lyrics条件相对不带这两者的分支差异，两分支仍共享相同ABC。这是增强文字条件的实验参数，不能保证准确歌词、固定音色或实际时长；仍需ASR加听审验证。

短片段可加 `--semantic-max-tokens 1200`，worker只向 `semantic_sampling` 传 `{"max_tokens":1200}`，其余采样参数不变。25Hz下这是约48秒的生成上限，可为约22秒片段保留两倍余量，减少CFG分支按prefix+max_tokens静态分配的KV显存。默认不指定时保持原生9000，旧manifest行为不变。上限必须至少200，以满足原生min_tokens；触顶产生truncated仍被拒收，不剪音频伪装成功。该配置进入manifest、输入identity、结果和原生config复用核验，不同上限不能复用音频。

## 本地渲染与断点

先查看实际命令：

```powershell
& $CpuPy -m tools.reference_studio render --experiment $Experiment --dry-run
```

只跑需要的臂，其他臂稍后续跑：

```powershell
& $CpuPy -m tools.reference_studio render --experiment $Experiment `
  --variant baseline --variant full
& $CpuPy -m tools.reference_studio render --experiment $Experiment `
  --variant sparse --variant arranged
```

控制进程用CPU Python，实际子进程固定使用 `vendor/YuE2/.venv/Scripts/python.exe -I -u`。默认 `--budget 12 --backend torch-eager --nar-query-chunk 1024`、AR offload，四臂同seed1338。可用 `--python` 指定YuE2运行环境，`--timeout-minutes` 默认40分钟。每臂独立加载/释放GPU，不同时并发。

每个 `renders/<variant>/` 包含 `state.json`、`render.log`、`audio.wav`、`result.json` 和原生YuE2 artifacts。只有子进程成功、结果未截断且音频哈希与请求指纹匹配才跳过。失败、参数变动或音频变化都会重跑；不会因“有WAV”便视为完成。`--force` 强制重跑指定臂，会替换该臂音频与日志。

### 显式复用相同输入的真实渲染

参考说明变了而实际ABC、歌词、style、seed完全相同时，可显式复用已完成推理，避免相同输入再跑一次：

```powershell
& $CpuPy -m tools.reference_studio reuse-renders `
  --from-experiment outputs/2026-10-19_quality_upgrade/reference/iron_rain_control `
  --experiment outputs/2026-10-19_quality_upgrade/reference/wildfire_transfer `
  --variant baseline --variant arranged
```

此操作先核对源与目标输入文件字节、seed、Python、budget、backend、query chunk和offload参数，再检查源state的完成状态与音频hash、原生request/config/score的hash与实际输入、ABC和semantic均未截断、本地YuE2代码与模型身份。模型验证会在CPU读取约8GB权重，不加载GPU。所有选定臂都通过才开始复制；任意臂仍在运行或失配就拒绝，不写complete。

目标音频保留原hash，`reuse_source/`保存原状态、结果和关键原生证据；目标state/result写明 `input_equivalent_reuse`、`inference_performed: false` 和完整来源。报告标注输入等价复用，不当作新推理或独立复验。不会替换目标已有渲染，也不允许把复用结果继续当成新的原始推理来源；重复调用可确认同一来源已复用。

## 试听和评分

```powershell
& $CpuPy -m tools.reference_studio report --experiment $Experiment
```

输出：

- `REPORT.md`：试听链接、状态与下一步。
- `report.json`：原始声学指标、基线差异、响度记录、人工评分。
- `listening/<variant>.wav`：所有已完成臂的同响度版本。
- `ratings.json`：对比度、律动、乐器进入清晰度、人声自然度、咬字、hook记忆度、混音、总体八项1–10分，默认为空。

试听使用单一线性增益，计算所有臂都能安全达到的共同RMS目标；遇到峰值限制会一起降低目标，避免一臂更响。这是mono RMS匹配，未宣称符合BS.1770 LUFS。分段报告按ABC推算时间，仍需听感校准实际音频边界。

`loudness_match.files`逐臂记录 `source_sha256` 与 `listening_sha256`，用于把归一化试听与真实完成源文件对应。分析解码显式按1/N做声道算术平均；`whole_mix`的峰值、RMS与crest直接测原WAV，避免ffmpeg默认等功率混音增益与降采样插值过冲误报源文件峰值。降采样mono的电平诊断另列在 `resampled_mono_level_diagnostics`。

报告同时给出真实与计划时长及相对误差。绝对相对误差超过10%时，标为 `technical_review_required`，该臂按符号边界的分段声学测量被禁用，不能把错误边界的能量差当成真实主副歌提升。整曲测量与原WAV仍保留，不做截短或拉伸来掩盖漂移。

先听sparse/full是否可辨，才判断Ins是否可控；再盲听baseline/arranged。至少验证两组相同条件不同seed，避免把一次随机变化当稳定控制。参考段落未听审、歌词退化、音色漂移或配器变化不可辨时，不应扩成批量整曲或宣称高分。程序不自动填分，也不自动提拔赢家。

## 固定人声的可控 MIDI 编曲素材

本地 YuE 的 ABC 条件不能强制真实音频按小节结束。需要精确鼓点、低音、和弦和弦乐进入时，先锁定已完成的干声文件，再单独安排四条 MIDI 轨。本步骤只准备素材，不调用模型、不生成乐器波形，不代表已完成伴奏或混音。

本轮配置已核实为86 BPM、4/4、D#m、8小节；和声取原ABC第一个chorus的逐小节标记 `D#m B F# C# D#m B F# C#`。SVS metadata本身没有和弦；这些标记不等于对旧生成伴奏的听审转录。两臂共享同一个comp v4干声SHA，准备时逐一核对其manifest、两个SVS父metadata和源ABC的SHA、调号、速度及和弦。原父文件保持不变。

从项目根目录运行：

```powershell
.\run_workflows.bat reference prepare-controlled `
  --config tools/reference_studio/examples/iron_rain_controlled_midi_v1.json `
  --vocal outputs/2026-10-19/experiments/voice/svs_comp_v4/comp_vocals.wav `
  --out-dir outputs/2026-10-19/experiments/reference/iron_rain_controlled_midi_v1
```

输出目录必须不存在或完全为空；需要第二版时换目录。独立入口是 `python -m tools.reference_studio.controlled_arrangement`，参数相同；Python接口是 `prepare_controlled(config: Path, vocal: Path, out_dir: Path) -> dict`。

素材包包括：

- `source/fixed_vocal.wav`：与comp v4逐字节相同的唯一干声副本，两臂都引用它；不再次生成、变调或拉伸人声。
- `sparse/arrangement.mid`、`expanded/arrangement.mid`：标准SMF format 1，每份含conductor和鼓、贝斯、和弦、弦乐四个独立乐器轨，480 PPQ。人声不被改写为MIDI音轨。
- 各臂`events.json`：四轨音符的起止tick、秒、音高、力度、通道与GM program。sparse使用低密度鼓、半小节根音与开放五度，弦乐第7小节进入；expanded增加鼓的八分律动、根音/五度贝斯和三和弦节奏，弦乐第3小节进入。GM program只表示预期角色，音源音色尚未验证。
- `markers.json`、`manifest.json`：每小节和声、精确进入/退出、统一END、参数与源文件/代码摘要、两臂相同的人声引用。`manifest.kind`为`controlled_arrangement_prepare`，状态固定为`prepared_not_rendered`，`rendering.performed`和`mix_available`均为false。

准确网格终点为32拍，即`960/43 ≈ 22.325581395`秒，最后tick为15360。MIDI标准速度字段只能存整数微秒，本例记录697674 µs/quarter，编码时长误差约−13.4微秒，manifest明确保留该差异。人声容器为22.326秒，与网格相差约0.419毫秒；这不代表实测静音长度，也不触发裁剪或时间拉伸。四轨在END发送note-off/all-notes-off/all-sound-off；未来实际音源的release尾音和音频同步仍需渲染后核对。

当前没有已验证的本地离线音源链，因此不提供假乐器试听。可把MIDI与同一干声导入DAW，加载真实音源后再做听审和混音；comp v4的歌词与音色也仍须人工确认。

## 验证

```powershell
& $CpuPy -m pytest tools/reference_studio/tests -q
```

测试覆盖真实Iron Rain计划的完整主副歌、不变Vocal/和弦/时值、第二组歌词定位、错误ABC拒绝、蓝图映射、静音与分轨对齐、同响度峰值约束、dry-run无GPU、音频hash断点与截断失败。可控编曲测试另用独立SMF解析器核对四轨音符/通道/结束标记、固定干声、和声来源及非空目录拒绝。GPU渲染和人工听感结果需另行记录。
