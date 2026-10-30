# 固定歌手与参考编曲工作流

2026-10-19 新增，两条流程共用本地 GPU，统一入口 `run_workflows.bat`。既有生成器及音频不覆盖，模型、参考音频与试听结果不随代码发布。

## 运行入口

在项目根目录执行：

```powershell
.\run_workflows.bat doctor
.\run_workflows.bat voice --help
.\run_workflows.bat reference --help
.\run_workflows.bat status
.\run_workflows.bat review
.\run_workflows.bat package --date 2026-10-19
```

统一入口记录每次命令、退出状态和日志，并用操作系统文件锁串行调度，防止这两个入口同时占用 GPU。直接调用各 Python 模块时，需要自行保持 GPU 串行。命令成功、文件可播放与主观质量验收分别记录。

## 固定歌手

固定 ID：`iron_rain_female_v1`。音色母本来自 2026-10-16 选中第4首 `04_iron_rain_raw.wav`，优先验证 4–11 秒干声。歌手档案保存原件、来源哈希、参考片段、F0、模型与参数。候选片段的统计不能替代音色试听。

完整操作和独立模型安装见 [voice_studio](../tools/voice_studio/README.md)。

### 2026-10-22：原生交付与 hybrid 实验

用户复听 19/20 的固定女声批次后判定高音/转音"变成机械音"，比 YuE2 原生更差。现在固定歌手线分两条：

- `voice native`：**默认交付**。把 YuE2 原生 44.1/48 kHz take 直接归档，不做换声、EQ、压缩或限幅。音色一致性只靠固定标签 + 种子 + 选定 anchor，不等于严格同一位歌手。
- `voice hybrid`：**实验路径**，不默认交付。全带 Demucs 干声 → 音区参考（`iron_rain_female_v2`）→ pyin 50 Hz F0 → SoulX-SVC → 24k merge → 11 kHz 以上从同一时间线干声恢复 → 44.1k 混音。必须带 `--highnote-qa`；`band_limited_24k` 或 `pitch_tracking_mismatch` 会把 `delivery_gate.deliverable` 置为 false。
- 旧 `cover`/`relyric` 仍可用于复现 legacy 24k 路径，但 manifest 标记 `deliverable=false`。

2026-10-22 小样（`outputs/2026-10-22_highnote_ab/`）显示：hybrid 的带宽/HF 比 legacy 好，但 02 的 1174 Hz 长音仍出现 F0 漂移和 1388 Hz 噪声段，未达到"不输给 native"的升格条件。听审入口见该目录 `REPORT.md`。

```powershell
.\run_workflows.bat voice native --source <YuE2 take.wav> --lyrics-file <lyrics.txt> --out-dir <dir> --title <slug> --execute
.\run_workflows.bat voice hybrid --workspace data/voice_studio --voice-id iron_rain_female_v2 --source <window.wav> --vocals <demucs vocals.wav> --instrumental <demucs no_vocals.wav> --job-id <id> --highnote-qa --execute
python -m tools.generation.highnote_ab --config outputs/2026-10-22_highnote_ab/ab_config.json
```

入口包含三种能力：

- `profile`：登记并提取固定音色。默认先写计划，`--execute` 执行真实音频处理。
- `cover`：目标歌曲分轨后执行 SVC，保持目标歌词和旋律，使用指定音色。
- `relyric`：使用已经唱出新歌词的导唱执行 SVC，或使用严格对齐元数据执行 SVS。只提供歌词文本不会被记为改词成功。

改词优先使用“新词的音素＋音符＋时值 → SVS → 固定参考音色”；已有准确导唱时再用 SVC。YuE2 也能尝试生成导唱，但本轮首个导唱实测没有正确唱出新词，因此没有送进换声后当成填词成品。新副歌在 `inputs/workflows/iron_rain_new_chorus.txt`。自动对齐、转录和参考分离均有误差，成品需检查漏词、高音身份漂移和接缝。

专属模型位于 `models/voice_studio`，推理解释器位于 `vendor/voice_studio/.venv`。此环境通过只读 `.pth` 复用现有 ACE 和音频分析库，另外安装 `transformers==4.41.2`、`tokenizers==0.19.1`，不改动既有环境。源码和公开权重固定 revision；音色母本始终使用本项目选定女声。

歌手档案长期保存在 `data/voice_studio/profiles`，已归档完整父原件，和临时交付目录分开。推理任务输出保存在 `data/voice_studio/jobs`，试听汇总复制真实音频到交付目录。

复用这位歌手翻唱自己的另一段音频：

```powershell
.\run_workflows.bat voice cover --workspace data/voice_studio --voice-id iron_rain_female_v1 --source D:/你的歌曲.wav --start 30 --seconds 20 --job-id your_song_sample_v1 --execute
```

先生成短样再扩展。`--seconds 0` 处理完整输入；参数、源文件或后端代码／依赖升级引起缓存标识变化时，换新的 `--job-id`。有已校验且时间对齐的干声／伴奏时，可通过 `--vocals` / `--instrumental` 直接使用，省去重复分轨。

2026-10-19 已用同一条链路完成首支整曲：`cover_no_white_flag_full_v1`（No White Flag，201.239 秒，18 个 12 秒分句）。技术检查通过（峰值 0.970、削波 0、掉音比 0.00432）；自动转写内容代理与原唱干声接近（字符相似度 0.653 vs 0.704），尾部副歌重复来自原曲本身。成品与逐行中英对照在日期目录 `outputs/2026-10-19/finished/`。**SoulX-SVC 工作在 24 kHz，人声高频上限 12 kHz**；另附的 44.1 kHz 混音只把 Demucs 原始伴奏接回，不恢复人声高频。音色身份、自然度和高音区仍需人工听审。

## 参考编曲

完整参数及控制边界见 [reference_studio](../tools/reference_studio/README.md)。

```powershell
.\run_workflows.bat reference init-annotations --reference human_refs/refs/hoyo-mix_wildfire.m4a --out inputs/workflows/wildfire_sections_draft.json
.\run_workflows.bat reference blueprint --reference human_refs/refs/hoyo-mix_wildfire.m4a --annotations inputs/workflows/wildfire_sections_draft.json --out outputs/2026-10-19_quality_upgrade/reference/wildfire_blueprint.json
.\run_workflows.bat reference propose-mapping --blueprint outputs/2026-10-19_quality_upgrade/reference/wildfire_blueprint.json --out outputs/2026-10-19_quality_upgrade/reference/wildfire_mapped_blueprint.json --source-sections section_01 section_02
```

段落边界首先由声学变化提出，保持 `section_XX` 及未审核状态。需要将参考区间映射到目标歌曲的 verse/chorus，并为目标提出稀疏或展开的器乐方案；源曲真实段落角色、乐器身份不根据频带能量臆测。源参考的音频只用于结构与声学分析。

`prepare` 从原 ABC 裁出完整连续段落，固定人声、和弦、时值、歌词和种子，生成四个对照计划：

| 版本 | 用途 |
| --- | --- |
| baseline | 原计划基准 |
| sparse | 器乐稀疏，验证 Ins 控制 |
| full | 器乐展开，验证 Ins 控制 |
| arranged | 按明确蓝图映射安排目标主副歌；无映射时仅为通用控制实验 |

Iron Rain 实验取第一主歌及紧接副歌，21 小节，58.605 秒，计划路径为 `outputs/2026-10-16_epic_hook/candidates/04_iron_rain_20261016-150430/04_iron_rain_c2_artifacts/score.abc`。本轮推荐继续从 `iron_rain_support` 组检查：缩短风格描述，统一为谱面86 BPM、D-sharp minor，并使用温和根音／五度伴奏；长提示词里的172 BPM、B minor及未出现的桥段、尾奏不再混入短片段。

```powershell
.\run_workflows.bat reference prepare --plan outputs/2026-10-16_epic_hook/candidates/04_iron_rain_20261016-150430/04_iron_rain_c2_artifacts/score.abc --lyrics inputs/2026-10-15_epic_en/04_iron_rain_lyrics.txt --style inputs/workflows/iron_rain_reference_style_short.txt --blueprint outputs/2026-10-19_quality_upgrade/reference/wildfire_mapped_blueprint.json --arranged-pattern support --seed 1237 --out-dir outputs/2026-10-19/experiments/reference/iron_rain_next
.\run_workflows.bat reference render --experiment outputs/2026-10-19/experiments/reference/iron_rain_next --variant baseline --variant arranged
.\run_workflows.bat reference report --experiment outputs/2026-10-19/experiments/reference/iron_rain_next
```

每个版本使用新的 GPU 进程，启用 12GiB 内存预算、CPU offload、torch-eager 和 NAR query chunk。失败、截断和缺失输出保留在记录中。`report` 将已完成版本按同一安全 RMS 做线性响度匹配，再输出试听与评分表。

YuE2 的 `Ins` 是器乐旋律声部，不能据此承诺逐件乐器、鼓型和混音均被精确控制。首次对照验证这条控制通路；如果实际听感不跟随计划，则精确配器应转到 MIDI/DAW 分轨。

实测也证明：外部ABC是生成条件，无法强制音频按小节准时结束。首轮密集伴奏版实际81.559秒，比计划长39.17%，保留为失败诊断；报告会将时长偏差超过10%的版本标为技术复核，并停用该版按符号边界的段落指标。不会裁掉尾部伪装合格。

## 如何判断提升

先比较主副歌对比、律动、乐器进入是否清晰，再检查人声自然度、歌词可懂度、钩子和混音。自动统计描述音频，不生成虚构的审美高分。`ratings.json` 在未听审时保持空值；只有听过的具体结果才进入下一轮扩展。

首轮原始工作目录为 `outputs/2026-10-19_quality_upgrade`，完整副本已收进 `outputs/2026-10-19/archive`；后续实验、QA、音频和源码快照集中在日期文件夹 `outputs/2026-10-19`，最终状态以该目录汇总及各任务 manifest 为准。

目前两个20秒跨曲换声样片已真实生成。《No White Flag》前后自动转写字符相似度0.947，词句保留较稳定；《Hammer and Hymn》为0.630，需要复核。这个统计描述内容保留，不能证明音色身份一致或音乐质量得分。用户当前没有耳机，已同意继续使用该女声参考，试听验收保持待定。

直接SVS新副歌也已生成，任务 `data/voice_studio/jobs/relyric_chorus_svs_v1`，22.326秒。4句目标词均近似恢复，自动转写相似度0.851，长度比1.171；前两句有错词／插词嫌疑，后两句较完整，不能声称逐字准确。该实验使用独立档案 `iron_rain_female_svs_v1`，参考音频与原固定歌手相同，补充近似音素对齐；旋律整体下移一个八度，保留原音区元数据。用法和复现命令见 Voice Studio README 的“本轮短SVS新词实验”。

第二版 `relyric_chorus_svs_v2` 仅提前 `sparks` 起音并合并同音槽，保持旋律时间线、总时长和词序；官方处理器在该起音的音素主体帧从0增加到14。实际ASR相似度0.870、长度比1.124，仍为4/4近似恢复，重复词减少；Bright/rising/on与第二句插词问题仍待精修。试听页保留两版，未自动批准声音身份或审美赢家。

温和support编曲版实际60.039秒，偏离58.605秒计划约2.45%，较首轮39.17%收敛；ASR目标句近似恢复9/9，仍有较多额外转写。符号段落还未与实际演唱精确对齐，声学变化不能单独证明编曲审美升级。本轮交付的是可运行、可对照、会暴露失败的新工作流和样片；高分听感与音色一致性尚未验收。

同日又把这套对照扩到 38 小节（主歌1+副歌1+主歌2+副歌2，计划 106.047 秒）：baseline 实际 106.639 秒（+0.558%，通过时长门槛），support arranged 实际 138.959 秒（+31.035%，`technical_review_required`）。arranged 的语义 token 从 2667 增至 3475，说明更密的伴奏会让模型多生成，外部 ABC 仍然只是条件、不是时钟。因此不把 support arranged 扩成整曲成品；精确配器应转 MIDI/DAW 分轨，或先解决计划与实际音频的对齐。报告：`outputs/2026-10-19/experiments/reference/iron_rain_long_support_v1/REPORT.md`。

首轮新增测试共85项通过（Voice Studio 34、Reference Studio 47、入口取消与试听追溯4），并实际执行了SVC、SVS与两组YuE2对照。QA路径：`voice/qa`、`voice/svs_qa` 和 `reference/iron_rain_support/qa`，均在本轮交付目录内。

`review` / `package` 将产出汇总到上海时区当日的 `outputs/YYYY-MM-DD`，也可显式指定 `--date`。2026-10-19 的统一入口为 `outputs/2026-10-19/index.html`，双击即可离线试听；音频在 `audio`，首轮完整产出、歌手档案和已完成任务副本在 `archive`，后续实验直接写到 `experiments`，自动检查在 `qa`。持久音色和任务原件仍保存在 `data/voice_studio`。日期集核对文件摘要、不覆盖已归档内容，试听记录导出携带具体音频摘要，不会自动上传或替用户打分。

新增 `tools/music_qa.py` 可对每次新音频运行统一CPU检查，记录原生峰值、空音频、非有限数、逐词替换／漏词／插词、逐行恢复、字符相似度及全部输入摘要。它不将同一段转写重复用于多句匹配，不向ASR提供目标词提示。自动结果只定位问题，不证明音乐性或声音身份。

```powershell
.\run_workflows.bat qa --audio data/voice_studio/jobs/relyric_chorus_svs_v3/converted_vocals.wav --lyrics inputs/workflows/iron_rain_new_chorus.txt --out-dir outputs/2026-10-19/qa/svs_v3
```

## 同日继续开发

按用户要求，统一交付目录为 `outputs/2026-10-19`。`review` / `package` 复制已完成音频、固定歌手档案、任务原件、日志、输入配置及本轮源码快照；可离线播放的路径全部位于日期目录内。既有快照逐文件校验，新旧源码和更新后的首轮报告分别保存为不同版本。失败任务只作为诊断归档，不进入正式候选卡片。未完成实验不自动展示。

新词v3修正 `rising`、`walk/on`、`find/my` 三处跨休止词界，保持旋律、休止、词序、总时长和固定参考不变。官方处理器的未分配帧从62降至52，零音素主体槽从1降至0。实际新音频22.326秒；同一套严格QA的总词编辑6→5，插词3→0，但第三句出现替换回归，因此保留v2/v3作为对照。

新增CPU逐句组合候选v4：用v3前段和v2后段，在 `Every` 尾部11.678792秒共同低能位置做20毫秒线性淡化；该处并非句间静音。总时长、采样点数不变，淡化窗外逐样本保持对应父片段。统一QA总编辑6→5→4，字符相似度0.906→0.925→0.944；仍有4处转写替换，未逐字验收。相同后段音频在不同整段上下文中会被ASR识别成不同文字，因此这些数值不等同声学质量分。候选及父来源保存在 `experiments/voice/svs_comp_v4`，具体复现见Voice Studio README；拼接工具拒绝向非空输出目录写入。

编曲新实验使用原词首副歌、seed1238、统一短style、CFG1.5、1200 semantic token上限，对照两臂只改变Ins。旋律条件版baseline实际23.159秒，但混音/干声ASR均有明显重复；arranged未正常输出结束标记，触顶截断并被拒收。[官方SheetSage2说明](https://huggingface.co/m-a-p/SheetSage2#melody-only-abc-for-covers)支持melody模式保留Vocal与Ins两声部，不能将失败归咎于非法多声部输入。保留和弦的full条件诊断实际生成完成，32.919秒，相对22.326秒计划偏长47.445%，仍未通过时长门槛；已停用符号边界分段测量，试听页明确标注待复核。最终状态以日期目录的 `development_summary.json` 和实验state为准。没有通过拉伸、裁掉尾部或增大上限来伪装通过。

full和声诊断的统一QA为24替换/0漏词/24插词，目标句近似恢复0/4，字符相似度0.241379；总编辑数受额外词数量影响，不能因低于baseline的62就声称歌词改善。它属于完成但未通过的诊断，不能扩成整曲。

精确编曲新增 `reference prepare-controlled`：在原副歌86 BPM、4/4、8小节和D#m / B / F# / C#两轮和声上，准备稀疏与展开两份标准MIDI。鼓、贝斯、和弦、弦乐分别保存音符事件与进出时间，两版引用同一份v4干声，来源摘要相同。该功能只准备工程素材，不调用生成模型、不产生乐器音频或混音；现有本地环境尚无经验证的MIDI音源渲染链。下一步选定音源后再检查实际起音、尾音与歌声对齐，不能因小节网格正确就宣称听感提升。

```powershell
.\run_workflows.bat reference prepare-controlled --config tools/reference_studio/examples/iron_rain_controlled_midi_v1.json --vocal outputs/2026-10-19/experiments/voice/svs_comp_v4/comp_vocals.wav --out-dir outputs/2026-10-19/experiments/reference/iron_rain_controlled_midi_v1
```

复现时使用新的输出目录。原和声来自本项目Iron Rain的引用谱面，不是对Wildfire原曲的精确分轨转写；两版为新制作的配器控制素材，尚未听审。

同日收尾完整套件132项通过：Voice Studio 48、Reference Studio 56、统一QA 14、交付与试听追溯13、进程取消1。真实推理、统一QA和MIDI准备另有任务原件及摘要，不把单元测试当作听感或音色验收。

下一步听审时优先比较固定母本、No White Flag翻唱及新副歌v2/v3/v4，并提供具体时间点。参考配器与音色的一致性仍由实际听感确认，扩展整曲和精确分轨编曲应基于这些反馈。
