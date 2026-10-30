# 固定歌手 Voice Studio

独立于已停用的 `tools/vocal_repair` 处理链。仅复用其 Python 的 CPU Demucs / pyworld 依赖，不调用旧的补救、RVC 或硬编码官方女声的脚本。第一位歌手为 16 日第 4 首《Iron Rain》原始生成音频，ID `iron_rain_female_v1`。

## 实际能力与验收边界

* `profile`：保留原音频、提取干声、自动挑选最多 3 个不重叠的活跃参考片段，生成 50 Hz F0 与音域摘要。可指定人工选择的起点。参考身份一旦准备好即冻结；重新选样需要新的版本 ID。准备成功状态为 `pending_listening`，`preparation_status=reference_ready`。
* `cover`：官方 SoulX-Singer-SVC 真实音频转换，使用档案自己的女声参考，保留目标歌歌词、节奏、旋律。默认不自动升降调。默认只生成 30 秒样片。
* `relyric --mode guide`：已经唱出新歌词的导唱干声 → 固定女声 SVC。可来自自己演唱、YuE2、SVS 等，必须提供新歌词文本；文本与声音是否一致仍须听审。不会把旧歌词音频改名当作新词合成。
* `relyric --mode svs`：官方 SoulX SVS 的真实适配器，需提供已校对的音素／音符／时长元数据，以及 reference.wav 对齐元数据。歌词文本单独不能定义逐字演唱节奏，不自动猜对齐。

自动质量检查仅检测音频长度、静音丢句风险和削波。输出状态为 `awaiting_listening` 或 `technical_review_required`，没有自动“95分”或音色一致性已验证的声称。需要逐句试听、跨歌比较后才能批准歌手档案与作品。

## 依赖与准备

主入口仅需 Python + NumPy 和 PATH 中的 FFmpeg。推理用独立 `vendor/voice_studio/.venv`，不修改已有音乐生成环境。

默认路径：

* 官方源码 `vendor/voice_studio/SoulX-Singer`。
* 新后端 Python `vendor/voice_studio/.venv/Scripts/python.exe`。
* SVC 权重 `models/voice_studio/SoulX-Singer/model-svc.pt`，2,793,965,154 字节。
* SVS 权重 `models/voice_studio/SoulX-Singer/model.pt`，2,818,092,278 字节。
* 本地 Whisper `models/voice_studio/whisper-base`，包含 config、preprocessor_config 和 model.safetensors/pytorch_model.bin。SVC 内容编码器需要它，worker 强制本地读取，推理不会临时联网下载。
* CPU 预处理 Python `tools/vocal_repair/.venv/Scripts/python.exe`，需 Demucs、pyworld、NumPy、SoundFile。也可通过 `--audio-python` 指定其他环境。

源码当前需要 `transformers==4.41.2` 的旧 Llama 接口（同时 `tokenizers==0.19.1`），不可直接用 ACE 环境的 4.57 版本。核心还用 PyTorch、torchaudio、omegaconf、accelerate、librosa、SoundFile 和 tqdm。无需安装官方包含 NeMo/FunASR/SageAttention 的全量 requirements 来运行 SVC。预处理先使用 CPU DIO + StoneMask；强气声、高音及伴奏泄漏处的 F0 仍需试听检查。

官方参考：[SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)、[模型](https://huggingface.co/Soul-AILab/SoulX-Singer)、[Whisper base](https://huggingface.co/openai/whisper-base)。YingMusic V1 官方默认 Linux flash-attention 路线，本工作流未把它包装成已经验证的 Windows 后端。[YingMusic 官方](https://github.com/GiantAILab/YingMusic-Singer)

可复现的资产锁保存在 `assets.py`：SoulX 源码提交 `81aeb3ae772c70093c3de74dc23c92d983801ae4`，权重提交 `40493ad90286056c7a9095035164434a79daa8c9`，Whisper 提交 `e37978b90ca9030d5170a5c07aadb050351a65bb`。`download_models.py` 固定这些 revision，检查大文件官方 LFS SHA-256、小文件 Git blob 摘要与精确字节数。推理前也验证官方 checkpoint SHA-256，文件存在本身不代表下载完整。

```powershell
# 在新建的Python3.11环境中安装，勿对已有生成环境执行。
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r tools/voice_studio/requirements-svc.txt
python -m tools.voice_studio.download_models
# 需要SVS时加入--include-svs；下载完可独立复核，不启动GPU。
python -m tools.voice_studio.download_models --include-svs --verify-only
```

这台电脑的实际部署采用新建 `.venv` 中 `.pth` 只读引用 ACE 与 vocal_repair 的现成包，并仅在新环境覆盖安装旧版 transformers/tokenizers；既有环境未修改。若独立重建 CPU 预处理环境，可安装 `requirements-audio.txt` 并通过 `--audio-python` 指定。下载脚本不会安装依赖、运行推理或改写已有生成环境。

## 命令（在仓库根目录运行）

正常使用推荐统一入口 `./run_workflows.bat voice <子命令和参数>`，它选择项目 Python、记录运行日志并持有工作流锁；下面模块形式也可用于代码排查，GPU 执行请通过统一入口调度。

```powershell
python -m tools.voice_studio doctor --probe

# 第一次登记并实际提取，默认CPU分轨；完整原件保留在档案目录。
python -m tools.voice_studio profile --workspace data/voice_studio --execute

# 如已有自己校验的干声，省去重复分轨。
python -m tools.voice_studio profile --voice-id iron_rain_female_v2 --vocals D:/path/clean_vocals.wav --reference-start 22 --reference-seconds 8 --execute

# 原词翻唱样片。--start / --seconds 同时裁切干声和伴奏。
python -m tools.voice_studio cover --workspace data/voice_studio --source D:/path/song.wav --start 30 --seconds 30 --execute

# 用新歌词生成的干声导唱转成固定女声。
python -m tools.voice_studio relyric --mode guide --guide D:/path/new_lyrics_guide_vocals.wav --lyrics-file D:/path/new_lyrics.txt --instrumental D:/path/new_arrangement.wav --seconds 30 --execute

# SoulX元数据中音素/时长/音高/类型必须与time对齐，单句不超过25秒。
# 先在新版本歌手档案中通过--prompt-metadata绑定reference.wav元数据。
python -m tools.voice_studio relyric --mode svs --voice-id iron_rain_female_v2 --source D:/path/melody_vocals.wav --score-json D:/path/aligned_new_lyrics.json --lyrics-file D:/path/new_lyrics.txt --seconds 0 --execute

# 不带--execute只创建可审阅任务计划；不启动GPU。
python -m tools.voice_studio cover --source D:/path/song.wav
```

显存由上层工作流统一调度：YuE2 / ACE / SoulX 不应并行加载在 3060 12GB 上。默认 CUDA FP16、32 steps、CFG 3、12 秒分句；低能量处切句，前后 180 ms 上下文加权接回，长度与位置严格保留。静音段输出静音，不调用模型。CPU profile 选择与测试不会加载 SoulX GPU。

## 输出、缓存与恢复

默认持久目录 `data/voice_studio/profiles/<voice-id>/` 保存原件、完整干声、参考片段、F0、候选片段与档案；避免清理 `outputs` 时误删固定歌手。`data/voice_studio/jobs/<command_hash>/` 保存任务 manifest、原件副本、歌词、分句、生成日志、每句缓存签名、`converted_raw_vocals.wav` 原始合成干声、`converted_vocals.wav` 电平匹配干声、可选 `review_mix.wav` 和技术 QA。使用已有干声但没有伴奏时仅输出干声，不伪造混音。所有命令可用 `--workspace` 选择其他根目录；同一位歌手的任务需使用同一 workspace。

生成干声默认用整段统一的常数增益匹配源干声有效演唱区域 RMS，匹配范围限制在 ±6 dB；太安静或不足 250 ms 的内容不向上放大，干声全局峰值留在 0.98 内。原始合成输出单独保留。混音使用匹配干声与伴奏直接相加，`--vocal-gain-db` 继续提供相对人声电平控制；如求和超过 0.95 峰值，仅对整段求和做一个全局常数衰减。不使用动态压缩或随时间变化的限幅。manifest 的 `vocal_level_match` 和 `mix_gain` 记录实际增益，便于比较音色而不被电平变化误导。

如先裁出 0–60 秒以加快 CPU 分轨，`--source` 必须使用与干声同长的裁切音频；`--source-manifest` 归档裁切区间和父原件摘要，`--parent-source` 同时归档完整父原始音频。不能把 176 秒原件和 60 秒干声直接配成一对。

缓存使用音频 SHA-256、参考档案、SoulX／Whisper 权重和配置摘要、后端源码、参数和歌词摘要；同时在目标解释器用 `importlib.metadata` 读取 torch、torchaudio、transformers、tokenizers、NumPy 版本，纳入缓存 key 和 manifest，不导入重型 torch。Whisper 必要文件在推理前按官方固定 revision 的 SHA-256／Git blob 摘要及字节数逐一验证。重复任务只复用摘要匹配的产物；同名不同音频不会误命中。失败始终写 `failed` 并保留日志和中间文件；同一任务再次执行可续用已验证的完成句。不会覆盖原作。独立 `--job-id` 可以用于人工命名，但不能复用给不同参数。

SVS 的 `score`／`melody` 控制模式也纳入缓存身份。新任务执行前将目标 score 保存为 job 内的 `target_metadata.json`，核对其摘要与 `inputs.score_sha256` 一致，worker 只读取该持久副本；外部临时文件后来修改或删除不会改变这份归档。本轮已完成的 `relyric_chorus_svs_v1`／`v2` 另按其原 manifest 记录的摘要添加了 `archived_target_metadata.json`：这是**执行完成后补充的证据备份**，历史 manifest、worker spec 和所记录旧源码摘要均未回写。

试听清单：参考与目标像同一位女声；逐句无漏字错字；高音气声不漂移；分句边界自然；人声与伴奏对齐。合成器生成成功 ≠ 已达到这些要求。

测试：`python -m pytest tools/voice_studio/tests -q`。覆盖时间线接回、错长拒绝、掉句探测、同名缓存污染、参考不可变、缺权重失败记录、仅歌词不能伪装改词等实际故障。

`doctor` 缺少必要权重、Whisper 文件或依赖时返回非零退出码，供自动化阻止继续执行；歌手档案的人工待试听状态不会被当作依赖故障。

## 本轮短 SVS 新词实验

`svs_score.py` 是本轮英文短样的 CPU 准备工具。它使用固定版本 CMU 英语发音词典，将 ARPAbet 写成官方 `en_` 音素格式；参考音频的实际歌词来自本地 faster-whisper small 的自动词时间戳，不强行套用生成前的原始歌词。ASR 未标完的持续尾音仅在 DIO 初始有声段支持时延长，所有推断与低置信词均写入 `preparation_manifest.json`。

当前目标专门采用《Iron Rain》第一段副歌的 **8 小节、4 行新英文歌词、86 BPM、22.326 秒**：每行分配两小节，再按 CMU 音节权重提出词与音符的对应。`note_type=1/2/3` 分别表达休止、新词音符与续唱；同词同音高的续唱可以合并，跨行连音继续保留前一词。原 ABC 绝对音区为 MIDI 73–83，而参考母本较低，因此另存仅整体下移一个八度的 MIDI 61–71 版本；它保留音级与旋律轮廓，不能称为原绝对音高复刻。

这套规则**不是通用中文对齐，也不是逐音素强制对齐**。未知词会明确报错；CMU 首选发音、ASR 词时序、四行平均占八小节及音符分配均需要验证。所有产物标为 `approximate_alignment`、`human_reviewed=false`。官方 DataProcessor 的 CPU 校验仅说明张量可供模型读取：本轮参考存在 20 个 PAD 未分配帧、目标存在 69 个，另有两处 `sparks` 的短音素帧提示，不能据此声称咬字或节奏已经通过。需要实际合成和内容检查。

本轮准备产物位于 `outputs/2026-10-19_quality_upgrade/voice/svs_preparation/`，独立歌手档案为持久目录中的 `iron_rain_female_svs_v1`。它使用与 SVC v1 相同的 4–11 秒参考音频/F0，另外绑定近似参考元数据和来源记录，没有改动 SVC v1。`source_duration_carrier.wav` 为原音频截取的精确 535824 样本、24 kHz 载体，只提供时长和粗技术 QA；新歌词与目标旋律来自 score 元数据。

```powershell
# 参考干声自动转写；只使用CPU，需在现有ASR环境中运行。
tools/asr/.venv/Scripts/python.exe -m tools.voice_studio.svs_score asr --prompt data/voice_studio/profiles/iron_rain_female_v1/reference.wav --asr-model PR_AIMusic_Extension/models/faster-whisper-small --out outputs/2026-10-19_quality_upgrade/voice/svs_preparation/prompt_asr.json

# CMU词典使用svs_score.CMU_URL中固定的提交，下载到下面的dictionary路径。
tools/vocal_repair/.venv/Scripts/python.exe -m tools.voice_studio.svs_score prepare --prompt data/voice_studio/profiles/iron_rain_female_v1/reference.wav --prompt-f0 data/voice_studio/profiles/iron_rain_female_v1/reference_f0.npy --asr-json outputs/2026-10-19_quality_upgrade/voice/svs_preparation/prompt_asr.json --score outputs/2026-10-19_quality_upgrade/voice/new_lyrics_guide_v2/plans/baseline.abc --lyrics outputs/2026-10-19_quality_upgrade/voice/new_lyrics_guide_v2/lyrics.txt --dictionary outputs/2026-10-19_quality_upgrade/voice/svs_preparation/cmudict.dict --phoneset vendor/voice_studio/SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json --out-dir outputs/2026-10-19_quality_upgrade/voice/svs_preparation

vendor/voice_studio/.venv/Scripts/python.exe -m tools.voice_studio.svs_score validate-cpu --out-dir outputs/2026-10-19_quality_upgrade/voice/svs_preparation --prompt data/voice_studio/profiles/iron_rain_female_v1/reference.wav --phoneset vendor/voice_studio/SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json --soulx-root vendor/voice_studio/SoulX-Singer

# 本轮已建立的独立SVS档案。重建时使用新的版本ID，避免改变既有参考。
./run_workflows.bat voice profile --voice-id iron_rain_female_svs_v1 --source outputs/2026-10-19_quality_upgrade/voice/preparation/04_iron_rain_first60_source.wav --vocals outputs/2026-10-19_quality_upgrade/voice/preparation/demucs/htdemucs/04_iron_rain_first60_source/vocals.wav --reference-start 4 --reference-seconds 7 --prompt-metadata outputs/2026-10-19_quality_upgrade/voice/svs_preparation/prompt_metadata.json --source-manifest data/voice_studio/profiles/iron_rain_female_v1/source_provenance.json --parent-source outputs/2026-10-16_epic_hook/selected/04_iron_rain_raw.wav --execute

# GPU命令须由上层调度，在其它生成模型释放显存、model.pt校验通过后运行。
./run_workflows.bat voice relyric --workspace data/voice_studio --voice-id iron_rain_female_svs_v1 --mode svs --source outputs/2026-10-19_quality_upgrade/voice/svs_preparation/source_duration_carrier.wav --score-json outputs/2026-10-19_quality_upgrade/voice/svs_preparation/target_octave_down_metadata.json --lyrics-file outputs/2026-10-19_quality_upgrade/voice/new_lyrics_guide_v2/lyrics.txt --guide-manifest outputs/2026-10-19_quality_upgrade/voice/svs_preparation/source_duration_carrier_manifest.json --seconds 0 --control score --steps 32 --cfg 3 --seed 1237 --job-id relyric_chorus_svs_v1 --execute

# v2只修正sparks首次起音的词界，在相同音高内提前并合并一格。
./run_workflows.bat voice relyric --workspace data/voice_studio --voice-id iron_rain_female_svs_v1 --mode svs --source outputs/2026-10-19_quality_upgrade/voice/svs_preparation/source_duration_carrier.wav --score-json outputs/2026-10-19_quality_upgrade/voice/svs_preparation/target_octave_down_v2_metadata.json --lyrics-file outputs/2026-10-19_quality_upgrade/voice/new_lyrics_guide_v2/lyrics.txt --guide-manifest outputs/2026-10-19_quality_upgrade/voice/svs_preparation/source_duration_carrier_manifest.json --seconds 0 --control score --steps 32 --cfg 3 --seed 1237 --job-id relyric_chorus_svs_v2 --execute
```

v2 的变更与验证单独保存在 `v2_alignment_patch.json`，保留 v1 摘要、原文件与原档案。`sparks` 起音从 0.872093 秒提前到 0.697674 秒，合并相邻 MIDI 70 同音槽；总时长、整个音高时间线和 27 个新词起音的词序／拼写保持一致。官方 CPU 处理器实际映射中，该槽从 8 帧且 0 个音素主体帧变为 17 帧且 14 个音素加分隔帧，全曲 PAD 从 69 降至 62。它仅修复一个结构缺陷，不能据此认定 Bright／rising 等所有咬字混淆已经改善，实际音频须另行验证。

### v3：按休止边界修正前两句

`svs_alignment.py` 在 `chorus-v3` 配方中只处理已观察到的三种断词：把 `rising` 起音移到休止后，休止前的短音归还 `sparks`；把休止后重复的 `walk` 槽分给 `on`；把休止后重复的 `find` 槽分给 `my`。它不改歌词、CMU 发音、参考档案、任何音高或休止区间和总时长。起音容量足够的 Bright、storm／I 不作猜测性发音替换，所以 Right、oh／where 等 ASR 疑点仍可能保留。

新实验资产集中于 `outputs/2026-10-19/experiments/voice/svs_v3/`，v2 保留。`v3_alignment_patch.json` 是冻结的配方与不变量快照；官方 CPU 验证另写 `v3_cpu_validation.json`。CPU 容量和断词结构改善不等于已经证明听感改善。

本轮官方 CPU 实测保持 1116 帧，PAD 从 62 降至 52，零音素主体槽从 1 降至 0；rising 的首次音素加分隔帧从 6 增至 12，on 从 6 增至 15。my 提前后首次槽从 15 降至 6 个主体帧，但仍完整覆盖发音串；这是消除 find 跨休止续唱的明确取舍，须在音频检查中验证。

```powershell
# 配方拒绝覆盖已有输出；复现请用新的实验目录。
tools/vocal_repair/.venv/Scripts/python.exe -m tools.voice_studio.svs_alignment chorus-v3 --source data/voice_studio/jobs/relyric_chorus_svs_v2/archived_target_metadata.json --out-dir outputs/2026-10-19/experiments/voice/svs_v3_reproduce --profile data/voice_studio/profiles/iron_rain_female_svs_v1/profile.json --asr-evidence outputs/2026-10-19/archive/initial_quality_upgrade/voice/svs_qa_v2/asr_segments.json

vendor/voice_studio/.venv/Scripts/python.exe -m tools.voice_studio.svs_alignment validate-cpu --source data/voice_studio/jobs/relyric_chorus_svs_v2/archived_target_metadata.json --target outputs/2026-10-19/experiments/voice/svs_v3_reproduce/target_octave_down_v3_metadata.json --soulx-root vendor/voice_studio/SoulX-Singer --phoneset vendor/voice_studio/SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json --out outputs/2026-10-19/experiments/voice/svs_v3_reproduce/v3_cpu_validation.json

./run_workflows.bat voice relyric --workspace data/voice_studio --voice-id iron_rain_female_svs_v1 --mode svs --source outputs/2026-10-19/archive/initial_quality_upgrade/voice/svs_preparation/source_duration_carrier.wav --score-json outputs/2026-10-19/experiments/voice/svs_v3_reproduce/target_octave_down_v3_metadata.json --lyrics-file outputs/2026-10-19/archive/initial_quality_upgrade/voice/new_lyrics_guide_v2/lyrics.txt --guide-manifest outputs/2026-10-19/experiments/voice/svs_v3_reproduce/v3_alignment_patch.json --seconds 0 --control score --steps 32 --cfg 3 --seed 1237 --job-id relyric_chorus_svs_v3_reproduce --execute
```

### CPU 逐句拼接候选 v4

`phrase_comp.py` 只接受帧数、采样率、声道布局及 score 音高／休止时间线一致的两份干声。它在指定范围寻找两份声音都通过 RMS 和峰值阈值的低能量窗，进行互补权重的线性短交叉淡化，保持总样本数与淡化窗之外的父音频逐样本一致；找不到合格窗口就拒绝拼接。

本轮第 2／3 句的符号边界约 11.16 秒，两份实际音频在那里仍然很响，因此唯一候选采用 **11.678792 秒、20 ms** 的共享低能量窗，RMS 约为 -38.04／-39.94 dBFS。这个位置在 Every 尾部：前两句与大部分 Every 来自 v3，Scar 及后续来自 v2，不能称为在精确句界整句切换。输出位于 `outputs/2026-10-19/experiments/voice/svs_comp_v4/`，manifest 记录双父 SHA、窗口与实际峰值，标记 `kind=phrase_comp`、`inference_performed=false`、`awaiting_listening`。

归档后的复现请使用上面的 `svs_alignment` 命令和下面的 CPU comp 命令，新增实验统一放在 `outputs/2026-10-19/`，每次选择新的输出目录；helper 拒绝覆盖非空目录。v4 是 Every 尾部的共同低能交接，并非句间静音。首次候选的历史 manifest 保持原样，不追补后来新增的源码 SHA 字段；新生成的 manifest 会记录当前 helper SHA 和参数。本轮 v2／v3／v4 的 ASR 总编辑代理为 6／5／4，v4 仍有 4 处替换；同一音频片段在不同上下文中也可能转写成不同文字，因此这些结果不表示逐字准确，也不是音色或听感评分。

```powershell
# 当前候选已保存。helper拒绝覆盖已有候选；复现需使用新的out-dir。
tools/vocal_repair/.venv/Scripts/python.exe -m tools.voice_studio.phrase_comp --early data/voice_studio/jobs/relyric_chorus_svs_v3/converted_vocals.wav --late data/voice_studio/jobs/relyric_chorus_svs_v2/converted_vocals.wav --early-score data/voice_studio/jobs/relyric_chorus_svs_v3/target_metadata.json --late-score data/voice_studio/jobs/relyric_chorus_svs_v2/archived_target_metadata.json --lyrics outputs/2026-10-19/archive/initial_quality_upgrade/voice/new_lyrics_guide_v2/lyrics.txt --out-dir outputs/2026-10-19/experiments/voice/svs_comp_v4_reproduce --boundary-note "Shared low-energy window near the tail of Every; not an inter-phrase silence."
```
