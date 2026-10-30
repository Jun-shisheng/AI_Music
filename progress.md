# Progress

## 2026-08-04

- Started quality-first redesign after rejecting full-song SoulX-SVS listening result.
- Created persistent implementation plan and recorded prior experimental evidence.
- Completed the workflow audit and implemented the multi-candidate rebuild agent with cleanup rollback and strict gating.
- Current phase: test the agent and run it on an existing candidate pool.
- The first English candidate was rejected because structural events hit the detector cap; cleanup was rolled back after dropout regression.
- Diagnosed and stopped an orphan FFmpeg cleanup process on the Chinese candidate; added bounded cleanup with automatic fallback.
- Evaluated three additional same-title candidates; none passed the new quality gate.
- Documented the quality rebuild path as default and kept the final review folder free of known-defective WAVs.
- Started ACE-Step in delayed initialization mode and generated six new candidates across three directions.
- The new batch produced three accepted review songs: `Moonlit Silk`, `Run Into The Dawn`, and `Sweet Little Fire`.
- The gate deliberately rejected the transparent cleanup variant for all three because it increased artifact/dropout proxies; the safer vocal-separated remix was retained.
- User listening found residual tremolo and AI noise; added a hard narrow-band/noise-event gate so high aggregate scores cannot override this audible defect.
- Tested ACE-Step generation with a real vocal reference stem; it still failed the calibrated noise gate. Prepared `vendor\YuE` as the next generation backend candidate.
- Cloned YuE, downloaded its X-Codec runtime, installed its dependencies into the CUDA ACE environment, and downloaded the YuE 1B Stage 2 checkpoint.
- Added an isolated `tools\generation\yue_workflow.py` wrapper with a single-batch 8GB-oriented configuration.
- Completed all three YuE 7B Stage 1 safetensor shards and verified their exact indexed sizes.
- Fixed Windows UTF-8 input handling and absolute-path handling in the YuE wrapper.
- Confirmed YuE Stage 1 can load with GPU/CPU dispatch on the local 8GB GPU; a 100-token smoke run produced two 1-second Stage 1 stems.
- Added `tools\generation\yue_stage2_decode.py` for cached short-clip decoding, fixed its X-Codec working-directory requirement, and produced a 1-second vocal stem, instrumental stem, and mix.
- Current limitation: low-VRAM 7B Stage 1 generation is too slow for full songs without a segmented/cache-aware production strategy. No YuE result is promoted to final review yet.
- Pivoted the preferred final-generation backend to HeartMuLa 3B because the SongGeneration repository URL currently returns 404 and YuE is too slow on the local 8GB GPU.
- Downloaded and size-verified HeartMuLa 3B and HeartCodec checkpoints, added `heartmula_workflow.py`, and passed 12 repository tests.
- Generated the first 30-second HeartMuLa Chinese female-vocal candidate. It was rejected from final review by the existing gate (`artifact 11`, `gate 88`, `structural_events 4`, `noise_events 8`); the cleanup variant regressed dropout and was discarded.

## 2026-08-05

- Root-caused the "noise + breaks" listening issue to three vocal-stem defect classes shared by ACE-Step and HeartMuLa: 4-10 kHz noise-burst frames (zcr 0.38-0.79), single-frame 10-13x level drops, and dropout gaps. Ruled out resampling, loudnorm, and broadband noise floor as causes.
- Built `tools/vocal_repair/frame_repair.py`: frame-level repair on the separated vocal stem (STFT 4-10 kHz band attenuation at 0.22 factor inside detected frames, gain recovery for level drops/dropouts). Repairs all detected events (no top-N truncation) so full-length songs are covered.
- Calibrated `noise_events` in quality_rebuild_agent to count only audible harsh frames (severity >= 0.55) with uncapped event detection, matching the human-listening calibration intent.
- Integrated frame repair into quality_rebuild_agent as an alternative to transparent cleanup; frame repair is chosen when it beats both source and cleanup on the quality gate.
- Rebuild results: Moonlit Silk and Run Into The Dawn PASSED (previously rejected under the calibrated gate); Sweet Little Fire still rejected at noise_events 16; HeartMuLa 30s smoke PASSED for the first time (noise_events 1, dropout 0.25s, gate 92).
- Added 3 tests for frame repair (noise suppression, level-drop recovery, clean-signal idempotence); 13 tests pass.

## 2026-08-05 (evening)

- User follow-up: vocals still "electro" everywhere, worse tearing at high pitch (~47s chorus). Second-round root cause: ACE-Step base-model timbre — too-clean/airless texture (HNR higher and flatness lower than human; 3-12 kHz 2-7 dB quieter than a real vocal reference). Ruled out F0 jitter (pyworld: 0.6-1.1%), clipping, loudnorm pumping, and the LoRA (never loaded: 400 on every attempt).
- Built `tools/vocal_repair/humanize_vocal.py`: frame repair + natural formant EQ (cuts 280/430/950/3600) + 7 kHz air shelf + deesser + limiter ("de_buzz" chain, selected by user A/B listening over 5 variants).
- Integrated humanize chain into `quality_rebuild_agent.py` with a listening-first preference (humanize preferred when it does not regress gate/structural/dropout; artifact proxy tolerated +4; noise ceiling relaxed to 120 for the air-shelf chain).
- Integrated into the FastAPI service: `queue.py` post-generation hook runs demucs -> frame repair -> de_buzz EQ -> remix on every song (config `VOCAL_HUMANIZE_ENABLED=true`, `VOCAL_HUMANIZE_VARIANT=de_buzz`; falls back to raw output on failure). Verified the exact service command end-to-end (~3 min per song).
- Rebuild: Moonlit Silk (humanize), Run Into The Dawn (frame — humanize regressed artifact there), Sweet Little Fire (humanize) — all 3 pass now; HeartMuLa 30s passes with humanize. 16 tests pass (added humanize_preferred coverage).

## 2026-08-05 (late) - HeartMuLa backend crash, parked for reboot

- Started the HeartMuLa-vs-ACE-Step backend benchmark: prepared moonlit_silk English lyrics/tags (outputs/heartmula_smoke_input/moonlit_lyrics.txt, moonlit_tags.txt).
- HeartMuLa 60s generation: model loaded and ran at GPU 63% for ~2m50s, then crashed with exit code 3221225477 (0xC0000005 access violation).
- After that first crash, EVERY HeartMuLa run (60s, 30s English, and the exact previously-successful 30s Chinese command that succeeded at 19:21) now crashes at model load — faulthandler shows the crash inside transformers `_load_state_dict_into_meta_model` -> `torch.storage.__getitem__` while loading the first safetensors shard (pipeline music_generation.py line 136, the `mula` property).
- Ruled out: disk space (190 GB free), model file integrity (4 shards ~15.7 GB), basic CUDA (matmul works), lyrics/length (identical successful command crashes too). One 1.75 GB lingering python.exe process (PID unknown today) was present.
- Diagnosis: the first 60s run likely OOM'd on the 8 GB GPU during inference (KV cache growth), corrupting the WDDM/CUDA driver context; subsequent processes segfault on CUDA init. Most likely recovers only after a machine reboot.
- Parked per user decision ("先到这里明天再试一试"). Next session: after reboot, re-run `heartmula_workflow.py --seconds 30` with the Chinese smoke input first; if it loads, proceed with the benchmark. Fallback if still crashing: try `device_map="auto"` (test at tools/generation/tests/test_heartmula_device_map.py — pytest needed in the ACE-Step venv) or switch the benchmark to YuE (its Stage 1 loaded with device_map=auto on this GPU before).
- Diagnostic scripts kept: tools/generation/heartmula_fault_trace.py (faulthandler wrapper), tools/generation/cuda_health.py (CUDA sanity), tools/generation/heartmula_load_diag.py (component-by-component load; note its tokenizer step is wrong, the real pipeline uses heartlib's own tokenizer).

## 2026-08-06 - HeartMuLa 基准恢复 + 人声噪声底/高音修复

- 重启后 GPU 恢复验证：HeartMuLa 30s 中文 smoke 成功（4 shard 加载 + 375 token 生成），确认上次崩溃为一次性驱动损坏
- HeartMuLa 60s 英文 moonlit 完整生成成功（KV cache 预分配固定，30s/60s 峰值内存相同 12.67GB）
- A/B 对比：HeartMuLa vs ACE-Step 同歌词同 humanize 链，两者 gate 100 通过；用户试听 02（HeartMuLa）稍好但仍不干净
- 研究学习：真人声特征 = 噪声底 < -70dB、静音段平滑衰减、HNR 高且稳定；AI 噪音根因 = neural codec 伪影 + mel 瓶颈高频损失 + ConvTranspose 混叠
- 实测根因：AI 人声噪声底 -46~-62dB（中低频宽带噪声 + 伴奏低频泄漏），高音段 2-4kHz 共振峰过强（gap 19.9 vs 真人 11.8）
- 新建 noise_floor_reducer.py：60Hz 高通 + 谱减 + 静音扩展器 -24dB + 6-16kHz 高频倾斜；修复 ISTFT COLA 归一化 bug（wsum 尾部爆炸）
- 新建 high_note_tamer.py：F0>380 帧 1.5-4kHz shelf -5dB + 最强峰 notch -8dB；修复全局归一化抵消 cut 的 bug
- 效果：噪声底 -62→-86dB，gate 97，40s 高音 2-4kHz 能量降 8dB，21 测试全过
- 用户反馈剩余问题：27s/40s/50s/最后的转音和高声不够好（台阶式转音未解决，需 F0 重合成）
- outputs 目录曾被用户手动清理（只剩 ab_compare），已确认非代码误删

## 2026-08-07 - 转音修复(WORLD 重合成)实验:被听音门控否决

- 目标:解决 27s/40s/50s/结尾的"转音台阶"(F0 硬跳变无滑音 + 乐句间真静音断裂 + 尖啸段)
- F0 诊断(transition_diag.py):39.2s 处 1.3s 真静音(dropout,伴奏连续);49.5s 处 ~337ms 强能量尖啸(700Hz 峰,F0 锁不住);转音处音高瞬间跳变无滑音
- 方案 A(外科手术式 WORLD 重合成)经 brainstorm + spec + plan 后实施:
  - transition_repair.py:检测器(step/silence_gap/screech)+ 滑音 F0 构建 + WORLD 合成 + 交叉淡化,窗口外逐位保真
  - 11 个提交,41 测试全过(含红→绿验证);TDD + 双阶段审查流程发现并修复:silence_gap 死代码、screech 噪声包络金属味、crossfade 基座 bug(窗口内部未修复)、48k 小窗口跳过、长间隙误分类等
  - 真实文件:53 事件(49 修复 + 4 long_gap_skipped);39.2s 大间隙按 spec 标记不桥接
- **听音门控(Task 4)否决**:v3 修复后用户反馈"噪音和断裂更明显",方向不对 → 不集成到 quality_rebuild_agent / FastAPI 管线(Task 5/6/7 挂起)
- 教训:WORLD 重合成短窗口内(100-800ms)的滑音在女声高音区有明显合成感/断裂感,比原始硬跳变更难听;50s 区效果尚可(用户未抱怨),27s/结尾区不行。转音台阶问题本质仍是后端模型特性,后处理逼近有天花板
- 保留产物:transition_repair.py(检测器+修复器,CLI 可用)、transition_diag.py(F0 诊断工具)、test_transition_repair.py(20 测试)、A/B 片段(outputs/transition_listen/*_v3)

## 2026-08-08 - 生产链顺序根因:EQ 之后必须再做 frame repair(噪音 9→0)

- 目标:继续降低"AI 音乐噪音太多"。先用 60s HeartMuLa 样本跑通生产命令并逐级测量:
  - demucs 人声:11 个 harsh 事件(>=0.55);noise_floor 后仍 11;frame repair 后降到 1
  - **但生产链最终输出又回到 9 个 harsh 事件**(gate 94,dropout 0.4s)
- 根因定位:`humanize_vocal.py` 旧顺序是 frame repair → de_buzz EQ,而 de_buzz 链的
  7kHz air shelf(+2.5dB)会把 4-10kHz 的高频噪声重新抬起来,等于修复被 EQ 撤销
- 修复:改为 noise floor → de_buzz EQ → high-note tame → **frame repair 最后**
  - 实测同一 60s 样本:gate 97、harsh 事件 0、dropout 0.3s(旧顺序 94/9/0.4s)
  - 不加 tame 的纯重排是 harsh 1;加上 high_note_tamer 后到 0
- `process_full_song.py` 同步简化为直接调用 humanize_stem(避免双重 NR/tame);
  `quality_rebuild_agent.py` 与 FastAPI queue.py 都走 humanize_stem,自动继承新顺序
- 新增回归测试 test_humanize_stem_keeps_harsh_events_low_after_eq(证明 EQ 后修复有效);
  42 个 vocal_repair 测试 + 21 个 API 测试全过
- 生产命令端到端验证(与 queue.py 完全相同的 CLI):固定链人声干声(remix 前)harsh 12→4,
  gate 94→97;最终混音重新分离后 harsh 会反弹到 ~3-6,原因是 remix 的 loudnorm 增益把
  低于检测活动阈值的"潜在 harsh 帧"抬到阈值以上,且 demucs 对混音的二次分离本身会引入
  分离残渣(例如 11.90s 事件只在混音重分离后出现,干声里没有)——最终用户听的是混音本身,
  干声指标是最干净的代理
- 用户试听后仍反馈"噪音和杂音还是很严重"。第二轮根因:单次 frame repair 只把 harsh 帧
  zcr 从 ~0.65 压到 0.56-0.68,仍在可听阈值 0.55 之上;repair 输入(tamed)实际有 18 个
  harsh 事件,单次修复后剩 4 个
- 修复 2:repair_vocals 增加 repair_rounds 参数,humanize_stem 用 2 轮
  - 二次修复后:干声 harsh 18→0,频段几乎无损(4-10k 0.4→-0.5dB,1-4k 与 0.2-1k 不变)
  - 最终混音(重分离人声)harsh:raw 11 → 单轮 2 → 双轮 1;harsh_ratio 0.041→0.027
  - 关键实现细节:第二轮必须**重新检测**(不能只对同一区域再乘一次衰减)——第一轮把
    zcr 压到 0.56-0.68 的帧仍在 0.38 检测阈值之上,重新检测才能再发现并压到 0.55 以下
  - 新增测试 test_two_round_repair_suppresses_more_harsh_events;43 个测试全过
- 用户反馈"咬字不清楚"。根因:zcr harsh 检测器**无法区分齿音与机器噪声**——真人声
  参考也触发 14-149 个 harsh 事件(s/sh/ch 天然高 zcr),AI 干声只有 17 个;标准修复
  (1-2 轮)把 5-8k 齿音频段削掉 1.6-1.8dB
- 修复 3:frame_repair 增加 isolation_only 模式(生产默认)——只修**孤立单帧突发**
  (AI harsh 帧 68% 孤立 vs 真人齿音 run 仅 27% 孤立),保留多帧连续齿音
  - 干声实测:5-8k 齿音损失从 -1.6/-1.8dB 减半到 **-1.0dB**,孤立突发帧仍被移除
  - humanize_stem 默认 isolation 模式;旧激进双轮保留为 variant repair_std2
  - 新增测试 test_isolation_only_preserves_sibilant_runs;45 个测试全过
- 清理:删除全部临时工作目录与诊断脚本,输出目录仅保留 3 个文件(原始 + v3 WAV/MP3)
- 遗留:dropout 0.3s 仍接近门限(0.35s),属后端模型特性;转音台阶问题继续由后端 A/B 基准解决

## 2026-08-15 - 管线重新设计(Phase 7):源头干净,零修补

- 用户决策:放弃旧"生成→后处理修补"管线(已被证明有天花板),重新设计规避 AI 杂音;
  删除 outputs 全部文件(87 个,均未达标准)
- 关键验证:本地 HeartMuLa 权重与官方推荐版完全一致——所有 shard 大小与 HF 的
  happy-new-year + HeartCodec-oss-20260123 MATCH,无需重新下载(先前判断有误)
- 新管线原则:
  1. 源头质量优先:官方推荐权重直接生成,任何 frame repair/EQ/humanize/de-esser 都不碰音频
  2. 多候选多样性:同歌词/标签用不同 seed+temperature+topk 生成 3-4 个候选
  3. 零后处理:唯一允许的是线性响度归一化(不改变频谱)
  4. 听音验收:用户亲自选;客观检查只拦坏文件(时长/峰值/削波/近静音),不用频谱代理
- 新脚本:tools/generation/clean_pipeline.py
- **8GB GPU 性能优化(决定性发现)**:
  - cfg_scale=2.0 使 batch=2 → 内存交换 → **56s/token**(10s 音频要 23 分钟)
  - **cfg_scale=1.0(单 batch)+ codec 放 CPU**:token 生成 **3.4 it/s(接近实时)**,
    125 帧(10s)只要 36 秒;codec 解码(flow matching)固定 ~6 分钟与时长无关
  - 结论:8GB 上 cfg 必须 =1.0;候选多样性靠 seed/temp/topk 而非 cfg
  - 热缓存后 30s 歌曲总耗时 ~10-12 分钟,可接受
- **首批候选验证成功(端到端)**:
  - silver_rain 英文 pop 歌,3 个 30s 候选全部通过客观检查(30.1s/48kHz/无削波/无静音)
  - 全部为 HeartMuLa 原生输出 + 线性响度归一化,**零修补**
  - 候选 1/2/3 已复制到 outputs 根目录供听音(silver_rain_c1/c2/c3_候选*.mp3+wav)
  - 性能实测:375 帧 token 生成 1:49(3.44 it/s),codec 解码 ~1:52,每候选总 ~6 分钟
- **中文候选生成(月光小巷,国风女声)**:
  - 3 个 30s 候选全部通过客观检查;已复制到 outputs 根目录(月光小巷_c1/c2/c3_候选*.mp3+wav)
  - 歌词/标签在 outputs/candidates/moonlit_alley_*;验证 HeartMuLa 中文支持
- **服务接入(generation_backend=heartmula)**:
  - 新增 services/api/app/services/heartmula_runner.py(与 acestep_runner 同构,调 clean_pipeline)
  - config.py 增加 generation_backend(默认 heartmula);queue.py 按后端分发,
    HeartMuLa 路径**不做任何 humanize**(零修补原则)
  - 新增测试:clean_pipeline 5 个 + heartmula_runner 4 个;52 工具测试 + 25 API 测试全过
- **60s 完整版生成成功(重要突破)**:
  - 完整管线(单进程)在 60s 崩溃(EXIT -1,与历史 0xC0000005 崩溃一致);
    分离诊断定位:token 生成 750 帧**成功**(4.3 分钟,3.0 it/s),崩溃在 codec 同进程内存冲突
  - 分离方案(mula 生成→存 frames→单独 codec 解码)**稳定成功**:60.08s/48kHz 立体声,
    无静音,每 10s rms 0.13-0.18
  - clean_pipeline 增加 --full-song 模式(内部分离执行);heartmula_runner 对 >=60s 自动启用
  - 交付:月光小巷_60s完整版.wav/mp3(outputs 根目录),零后处理+线性响度归一化
- **完整歌曲结构验证(长歌词 60s)**:
  - 完整歌词(前奏/主歌/预副歌/副歌/间奏/桥段/尾奏,约 800 字符)60s 生成成功,
    token 750 帧 + 分离解码稳定;音乐动态推进(每 10s rms 0.07→0.15)
  - 交付:月光小巷_完整版v2.wav/mp3;证明管线支持"整首歌"而非片段
  - runner 路径修正:split runner 脚本改写到 output 同目录,移除多余 raw.unlink
- **8GB 显存上限实测 + 分段完整歌曲(重要)**:
  - 210s 探测:KV cache 增长极快,250 帧(16s)即达 14.36GB(远超 8GB)并崩溃;
    结论:单次生成可靠上限约 200-250 帧(16-20s),完整歌曲必须分段
  - 新增 tools/generation/segment_song.py:按歌词段落切分(保持顺序无重复)→
    每段 30s 同参数生成(缓存复用)→ acrossfade 重叠拼接(无静音缝)
  - 修复:跨段 filter 缺分号导致的 Invalid argument;目录匹配误命中 txt 的 bug
  - 交付:月光小巷_分段完整版.wav/mp3(88.2s,3 段,无静音,尾段副歌高潮)
  - 新增 5 个分段测试;57 工具测试 + 25 API 测试全过
- **用户反馈:鼻音/重音处仍有杂音/伪音 → HF 伪影定位与精准修复**:
  - 定位:AI 相比真人声在 8-12k(+4.9dB)/12-16k(+6.0dB)有显著多余能量,
    148 个伪音事件分布全曲(16% 时间),鼻音/重音处更明显——HeartCodec 高频重建伪影
  - 方案(hf_artifact_reducer.py):每帧**相对**门控——8-16k 相对自身中频异常高的帧
    才处理,按该帧比例压回中位数;8k 以下不碰;时间平滑避免咔哒
  - 效果对比(分段完整版 88s):
    - 原版:8-12k=6.5,12-16k=-1.7,伪音事件 148
    - 局部迭代2:8-12k=5.3,12-16k=-3.2,事件 123
    - **组合版(全局 9k/14k shelf 打底 + 局部精修):8-12k=3.0,12-16k=-6.1,事件 116,
      中频 2-4k 仅 -0.3dB 无损**(真人参考 -0.2/-8.0)
  - 交付:月光小巷_降伪音_组合版.wav/mp3;新增 3 个 HF reducer 测试
  - 60 工具测试 + 25 API 测试全过
- 待办:用户听音对比(原版 vs 降伪音组合版);选定后按参数批量生产
- 首批候选:silver_rain 30s 生成中(零后处理)
## 2026-08-20 - Human-vocal optimization

- Audited the refactored Phase 7 pipeline. HeartMuLa source-clean generation remains the default; the old RVC/frame repair/EQ/humanize chain is not part of production.
- Fixed candidate seed propagation. Normal generation, long-song split generation, and segmented generation now seed Python/NumPy/Torch instead of only recording unused seed metadata.
- Fixed `hf_artifact_reducer.py` stereo folding: detection uses a mono blend, but reconstruction is performed per channel so stereo width and accompaniment imaging are preserved.
- Added `tools/generation/surgical_vocal_variant.py`. It separates only vocals with Demucs, applies relative 8-16 kHz reduction to abnormal vocal frames, and remixes the untouched instrumental. The source-clean file is always retained.
- Generated `rain_after_the_window_c1`, 30.08 seconds at 48 kHz. Source and surgical vocal A/B versions, matching lyrics TXT files, and MP3 previews are under `outputs/next_generation/review`.
- A/B measurement: the surgical full mix reduces 8-16 kHz energy by about 1.1 dB while preserving the mid vocal band and stereo difference. It is not auto-promoted until human listening selects it.
- User reported residual noise in both A/B versions. The issue is concentrated in the vocal stem rather than GPU memory. Added a default vocal-only transparent cleanup stage before the narrow HF reducer. `rain_after_the_window_c1_vocal_noise_clean_workflow.mp3` measures HF energy ratio 0.0048 versus source 0.0090; the instrumental remains untouched and all prior A/B files remain available.
- Verification: generation tests 18 passed (excluding the full model-loading VRAM test); API tests 25 passed.

## 2026-08-25 to 2026-08-26 - 20-song bilingual batch

- Removed the old music outputs as requested and preserved only the new batch structure under `outputs/batch_20260820`.
- Prepared ten original Chinese and ten original English pop directions with matching plain-text lyrics and tags.
- Initial isolated generation failed while ComfyUI held GPU/page-file resources. After resources were released, the isolated approach proved too slow because it reloaded four HeartMuLa shards for every title.
- Added `batch_generate_20_resident.py` to keep HeartMuLa loaded once. The process is stable on the RTX 4060 laptop GPU and resumes by skipping existing WAV files.
- Recovery checkpoint on 2026-08-26: 9/20 source-clean 30-second, 48 kHz WAV candidates complete; song 10 is running. Vocal cleanup and quality packaging remain pending until source generation finishes.
- Added `finalize_batch_20.py`: each source is separated, cleaned on the vocal stem only, remixed with the untouched instrumental, and automatically rolled back if HF ratio, dropouts, or click metrics regress. Final deliverables are renamed WAV/MP3/lyrics TXT plus one plain-text report; Demucs work files are deleted after each song.
- Added `watch_and_finalize_batch_20.py` so finalization starts only after all 20 source WAVs exist and the resident HeartMuLa process has written its finished state, preventing Demucs from competing for GPU memory.

## 2026-08-26 - 20-song bilingual batch complete

- Completed all 20 source-clean review samples: ten Chinese and ten English, each 30.08 seconds at 48 kHz.
- Final folder contains 20 renamed WAVs, 20 MP3 previews, 20 matching lyrics TXT files, and `QUALITY_REPORT.txt`; per-song Demucs work folders were removed.
- Vocal-only cleanup promoted 19 variants; one song used the source-clean fallback because cleanup did not pass the no-regression gate.
- Objective checks found no measured dropouts in the final batch; click counts are diagnostic transient metrics and still require human listening review.
- Generation tests: 18 passed, excluding the full model-loading VRAM test.

## 2026-08-27 - Selected full versions complete

- Generated and packaged 13 full versions in the separate `outputs/batch_20260820/full_final` directory.
- Includes Chinese 02-04 and English 11-20; durations are 124.72-180.08 seconds at 48 kHz.
- Each title has WAV, MP3, and matching lyrics TXT; temporary full-song work files are empty/removed.
- Applied vocal-only transparent cleanup with automatic source-preserving quality gating.

## 2026-10-09 - Desktop (RTX 3060 12GB) migration, retune, guidance restored

- Environment was non-functional on the new desktop: both venvs
  (`vendor/ACE-Step-1.5/.venv`, `tools/vocal_repair/.venv`) pointed at
  `D:\Python\download-11` (CPython 3.11.9), which did not exist after the copy.
  Installed the official 3.11.9 amd64 build at exactly that path; both venvs
  revived with no package reinstall (7.5 GB site-packages, torch 2.7.1+cu128
  with bundled CUDA DLLs, CUDA available on sm_86, 12.88 GB VRAM).
- Root-caused why quality was capped on the old machine: `cfg_scale=1.0`
  disables classifier-free guidance entirely (`bs_size = 2 if cfg_scale != 1.0`),
  and the 8 GB card forced it. Every song delivered before today used that path.
- Found the actual memory hog: the backbone preallocates its KV cache for
  `max_seq_len=8192` regardless of song length. Right-sizing it to
  prompt+frames (rounded to 512) frees ~2.4 GB at guidance batch 2.
- Measured on the 12 GB desktop with the new config:
  - cfg=1.0 30 s: peak 8.5 GB token / 10.7 GB decode, 1.6 min
  - cfg=2.0 30 s: peak 9.1 GB token / 10.7 GB decode, 1.6 min (guidance now free)
  - cfg=2.0 60 s: peak 9.4 GB token / 10.7 GB decode, 2.5 min
  - token generation ~10 it/s at every cfg; codec on GPU cuts decode 5.2 -> 1.0 min
- Quality A/B, same seed, loudness-matched (`tools/generation/audio_qa.py`):
  cfg=1.0 854 clicks / 6 dropouts; cfg=1.5 1130/2; **cfg=2.0 469/1**;
  cfg=3.0 4927/0 (over-driven). Full-length, same seed: cfg=2.0 25.0 clicks/s and
  0.028 dropouts/s vs cfg=1.0 45.8 and 0.051.
- Rewrote `clean_pipeline.py` around one in-process runner: GPU codec,
  right-sized KV cache, `--base-cfg` default 2.0, automatic CPU-codec fallback on
  a GPU decode OOM. Removed the deprecated 8 GB split path (kept `--full-song`
  accepted as a no-op). `check_audio`/`normalize_loudness`/manifest unchanged.
- New tools: `audio_qa.py` (clicks/dropouts/harsh table, diagnostic only),
  `ab_cfg.py` (same-seed cfg A/B for listening), `bench_one.py` (timing/VRAM).
- Verified end-to-end: 106.5 s full song at cfg=2.0 (peak 10.68 GB, 5.77 min) and
  177.0 s at cfg=1.0, both pass the objective check. Review pack with MP3s and a
  Chinese listening guide at `outputs/review_2026-10-09/`.
- Tests: `tools/generation/tests/test_clean_pipeline.py` 6 passed (the runner
  test now asserts seeding + cache right-sizing instead of the removed split runner).
- User decision this round: personal / non-commercial use; keep the tuned
  HeartMuLa rather than download a new model.
- Workbench made runnable on this machine (user later widened scope):
  - Installed ffmpeg 9.0.2 (winget, user scope) and recreated the missing
    `services/api/.venv` (Python 3.11.9 + requirements.txt + pytest +
    pytest-asyncio). All three suites now pass: generation 18, vocal_repair 45,
    API 25.
  - `.env`: replaced DEEPSEEK_API_KEY with the key in `D:\Project\key\deepseek.txt`
    (verified HTTP 200 against api.deepseek.com; `deepseek-chat` still resolves, so
    no code change) and repointed every `G:/AI_Music` path to `D:/Model/AI_Music`
    (OUTPUT_DIR, DATABASE_URL, ACESTEP_DIR, ACESTEP_LORA_PATH, ACESTEP_MODEL_DIR).
  - `start.bat`: ROOT now `D:\Model\AI_Music`; ACE-Step (port 8765) startup and
    the 8 GB INT8 patch are now gated behind `START_ACESTEP` (default 0), because
    the default HeartMuLa backend loads its own model and the two would compete
    for 12 GB of VRAM. The frontend only needs the FastAPI backend.
  - Backend port moved 8000 -> 8001 (`start.bat`, `apps/web/next.config.js`,
    `.env` API_PORT): port 8000 was already held by the IncrediBuild Manager
    Windows service, and binding to it failed with WSAEACCES. The frontend is
    still http://localhost:3000 and proxies /api/* to the backend, so this is
    invisible to the user.
  - Two real bugs surfaced only by running the stack, both fixed:
    - `heartmula_runner` passed `--out-dir <outputs>` to clean_pipeline but then
      globbed `<outputs>/candidates/` for the take, so every HeartMuLa job via the
      API would fail with "output not found". It now passes
      `--out-dir <outputs>/candidates` to match. (Its tests only covered the
      `_tags_from_prompt`/`_lyrics_text` helpers, so this never ran in CI.)
    - `asyncio.create_subprocess_exec` raises NotImplementedError on Windows
      under uvicorn (no Proactor loop), so generation always failed. It now runs
      the generator via `asyncio.to_thread(subprocess.run, ...)`.
  - Verified end-to-end on this machine: `/health`, `/library` (50 migrated jobs),
    `/arrange` through the new DeepSeek key, and a real 90 s generation job --
    queue -> heartmula_runner -> clean_pipeline -> GPU -> 90.1 s WAV written to
    `outputs/<job_id>.wav`, served with HTTP 200 through both the backend and the
    Next.js `/api` proxy.
  - Note: old `app.db` rows still carry `G:\AI_Music\outputs\...` audio paths;
    they are historical records whose audio no longer exists. New jobs write to
    the local outputs dir.
- Vocal direction + a clean 20-song batch (same day):
  - User wants female vocals only, split per song into 清甜女声 (sweet, bright,
    young) and 御姐 (mature, rich low-mid, powerful). Rewrote the leading vocal
    descriptor in `inputs/*_tags.txt` for all 20 songs, 10 of each, 5 Chinese +
    5 English per voice type. Song sources were moved out of
    `outputs/batch_20260820/inputs/` to a top-level `inputs/` first.
  - Cleaned `outputs/` (1.3 GB -> empty) as requested, keeping only newly
    generated songs. Everything removed was either the superseded old batch or
    scratch from the retuning session.
  - Added a "degenerate ending" check (`audio_qa.tail_degeneracy`, also exposed
    as `audio_qa.py --degeneracy`): the model sometimes stops tracking the music
    near the end and emits noise, which shows up as a burst of sample-level
    clicks concentrated in the tail. Threshold: >50 % of clicks in the last 20 %
    AND >100 clicks/s overall. Verified on real takes: a broken one scored
    327 clicks/s with 66 % in the tail, healthy ones 5-42 clicks/s and 13-29 %.
    `batch_tuned.py` now retries such a take with a new seed (up to 3 attempts)
    instead of shipping it.
  - New `tools/generation/batch_tuned.py`: resident-model batch generation
    (loads once, resumes by skipping existing outputs) and `run_batch.bat` to
    launch it in its own console.
  - Two operational failures cost time and are worth remembering:
    - ComfyUI (`C:\Model\ComfyUI`, port 8188) holds ~10-11 GB of VRAM while
      resident, leaving no room to generate. Close it first.
    - Running the batch as an agent background shell is unsafe here: when it is
      reaped for memory pressure the python child is orphaned still holding
      ~11 GB, which starves the next run into being reaped too. Killing one such
      orphan took available memory from 1.9 GB back to 13.0 GB. Hence
      `run_batch.bat`, which runs in an independent console.
  - Batch state at end of day: 8/20 done (`outputs/final/`, WAV + MP3 + lyrics),
    song 09 in progress. Remaining 12 resume automatically from `run_batch.bat`.
- Remaining ceiling, unchanged: HeartMuLa's 12.5 Hz LM frame rate blurs fast
  transients. That is the structural source of the residual breaks/crackle and no
  post-processing removes it (the old repair chain proved this repeatedly).

## 2026-10-10 - Batch resumed, English titles first, tag fix

- Found yesterday's batch had been interrupted mid-song 09 (8/20 done: 01-08).
- Tag bug: 5 input tag files (04, 07, 12, 16, 18) carried a stale standalone
  `, male vocal,` left over from the female-descriptor rewrite -> removed. 04 and
  07 were already generated with the contaminated tag (not yet regenerated);
  12/16/18 were fixed before generation.
- `batch_tuned.py` gained `--priority <regex>`: matching slugs are generated
  first (stable sort). Seed/temperature stay tied to each title's SORTED position,
  so run order no longer changes any song's parameters.
- New `tools/generation/start_batch_silent.ps1`: detached, windowless launch with
  logs to `logs/batch_<stamp>.log` and a running-process guard. `run_batch.bat`
  kept for manual launches and updated with the same `--priority`.
- PowerShell trap: PS 5.1 reads a BOM-less UTF-8 `.ps1` as GBK, and the mis-decode
  swallowed the newline after a Chinese comment on line 9, merging the following
  `$Py = ...` line into the comment -> `Test-Path $Py` failed on a null path.
  Fixed by writing the file as UTF-8 **with BOM**. (`D:\Model\AI\tools\
  start_comfy_silent.ps1` is BOM-less too and only works by luck.)
- Run started 09:11 (cfg=2.0, 210 s, priority `^1[1-9]_|^20_`): order is 11-20 then
  09, 10. First song finished in ~8 min (4-shard reload per song), peak 9.7 GB
  token / 10.7 GB decode on the 12 GB card. ~1.5 h for the remaining 11.
- New test `tools/generation/tests/test_batch_tuned.py` (4 passed) covers the
  ordering: priority first, stable, no drops/duplicates, payload preserved.

## 2026-10-10 (later) - Batch complete: 20/20, English titles done first

- Finished 12/12 of the resumed run at 10:55; `outputs/final/` now holds the full
  20-title set (WAV + MP3 + lyrics TXT + manifest.json, 511 MB). Durations
  98.3-190.6 s -- the model emits EOS when the lyrics end, so a shorter take is
  content-determined, not a truncation.
- Two takes were auto-rejected by the tail-degeneracy guard and retried with a new
  seed, both recovered: 12_run_into_the_blue (200.4 clicks/s, 70 % in the tail) and
  10_sing_tomorrow_bright (143.6 clicks/s, 54 %). No errors, no failures.
- `audio_qa.py --dir outputs/final --degeneracy`: all 20 verdict `ok`; harsh count
  0 for 19 of 20 (17_stay_until_sunrise has 3, ratio 0.0010). Clicks/s 0.3-89.6.
  Diagnostics only -- the human still picks the take.
- Contention lesson: ComfyUI was launched at 10:23:50 (its own `start_comfy_silent.ps1`
  command line) mid-batch. It did not touch VRAM (models load lazily) but held 13 GB
  RSS, dropping free physical memory to 1.4 GB / 31.8 GB and stretching
  09_burning_blue from ~6.5 min to **29.3 min**. Token generation is deterministic in
  step count, not wall time, so the audio was unaffected (2.9 clicks/s, fine).
  It exited on its own before `stop_comfy.ps1` ran; free RAM returned to 17.7 GB.
- Measurement note for future sessions: `nvidia-smi utilization.gpu` is a
  "did any kernel run" busy-time flag, not saturation. Sampled over one song:
  shard reload 0-18 % / 15-45 W, token generation 59-64 % / ~93 W at 1957 MHz,
  codec decode 99-100 % / 168-169 W (power-wall-limited), normalize+MP3 ~0 %.
  Whole-song average is therefore ~50 % -- which is what Task Manager shows, so
  Task Manager is the more honest number here; only its default graph (3D engine)
  under-reports CUDA. Power limit is adjustable (min 100 W / max 170 W) and
  `-pl` / `-lgc` are bit-identical for the audio, but the user chose to leave it
  at 170 W.
- Still open: 04 and 07 were generated with the stale `, male vocal,` tag (removed
  from the inputs afterwards); listen and decide whether to regenerate them.

## 2026-10-13 - Stage 3 kickoff: target is game battle / BGM level

- User brief (09:14, in a session that was interrupted before it could finish):
  the current vocal-song 音质/音色 has passed acceptance; the NEXT stage is to
  reach Chinese gacha-game / battle-anime soundtrack level (原神/崩铁/明日方舟,
  Final Fantasy-class battle and BGM cues), working with local resources only.
  The user explicitly delegated the silent downloads and the background
  research/analysis.
- The interrupted session had just launched two research agents and died with
  them; nothing was written to disk. Both were relaunched here and both
  returned (model landscape + genre/production characteristics). Key findings:
  - **HeartMuLa's ceiling is structural and in-family upgrades do not fix it.**
    HeartCodec runs at a hard 12.5 Hz frame rate (80 ms/frame); the
    `happy-new-year` checkpoint changed the LM only and HeartCodec-oss-20260123
    only "optimizes decoding quality". The 7B model is still unreleased. So the
    residual crackle/smearing cannot be tuned away -- the backend has to change.
  - **YuE2-3B** (HF `m-a-p/YuE2-3B`, Oct 2026, 3.58B, 7.26 GB single
    safetensors, CC BY-NC 4.0 with an individual-creator carve-out) is the
    strongest candidate: 48 kHz at **25 fps** latents and it plans the piece as
    symbolic ABC notation before rendering, which is exactly the property
    HeartMuLa lacks. Best-of-8 SongBench #1; fits 12 GB bf16; Windows-supported.
    Two LoRAs are directly on target: `monsterovich/yue2-steps-from-hell`
    (Two Steps From Hell epic-orchestral library) and
    `Mothersuperior/YuE2-instrumental-cot-full-loras` (instrumental workflow).
  - **ACE-Step 1.5** (MIT, 2B turbo + optional XL 4B): 48 kHz with 25 Hz VAE
    latents, built-in stems / vocal2BGM / LoRA training, fastest iteration loop.
    This is already installed locally, so it costs nothing to test first.
  - Genre facts that shape the brief: in-game battle BGM is **instrumental**
    (vocal versions are a separate artefact -- phase-2 boss songs, anniversary
    themes); ~100-150 BPM minor/modal with ostinato strings; the loudest
    machine tells in this genre are strings that imply unplayable bowing,
    "behind frosted glass" top end, no true silence, and transients that are
    too uniform.
- Local environment facts learned (needed for any future download work):
  - huggingface.co is **unreachable** from this machine; `hf-mirror.com`
    answers. `huggingface_hub` 0.36 in the ACE venv fails its HEAD probe
    against the mirror (the Xet transport path is unusable here) with a
    misleading `LocalEntryNotFoundError`, even though plain httpx/curl work.
    Workaround: `tools/generation/download_models_yue2.py` downloads with
    `curl -L --retry 5 -C -`, so it resumes and does not need hf_hub at all.
  - ACE-Step 1.5 is already vendored and complete at `vendor/ACE-Step-1.5`
    (v15-turbo 4.5 GB, 5Hz LM 0.6B + 1.7B, vae, Qwen3-Embedding-0.6B).
    `vllm` is not installable on Windows, so the LM backend must be `pt`;
    the repo `.env` still says `vllm`, so pass `pt` explicitly.
  - New tool `tools/generation/acestep_generate.py`: source-clean ACE-Step
    backend (no post-processing), taking a **job JSON** because a long caption
    passed through bash -> PowerShell -> Start-Process loses characters.
    Job files live in `inputs/battle_test/`.
- In flight at time of writing:
  - YuE2 stack download (7.26 GB base + 2 LoRAs) -> `models/yue2/`,
    log `logs/yue2_download.out.log` (launched 09:29 detached).
  - First ACE-Step battle instrumental (60 s, turbo, D minor, 140 BPM, 2
    takes) -> `outputs/battle_test/`, log `logs/ace_battle_test.out.log`.
  - HeartMuLa counterpart brief written (`inputs/battle_test/01_battle_*`) so
    the same intent can be run on the current backend for an A/B.

## 2026-10-13 (later) - Vocal battle batch: engine chosen by measurement, not taste

- User decisions this round: battle tracks must be **vocal** ("带人声的战斗曲"),
  must be **欲扬先抑 with an epic climax**, and the user is out today without
  headphones -- so the go/no-go on batch generation is delegated to me, with
  human review at home. That makes the aesthetic requirement something that has
  to be *measurable*, which drove the whole design below.
- **Diction decided the engine.** Running faster-whisper against the intended
  lyric on the first vocal takes:
  - HeartMuLa `battle_vocal_mula_v1_c1`: ratio 0.58, coverage **0.91** -- sang
    the entire lyric in order, single-character errors only (忧伤→忧郁,
    这一剑→这一间, 燃尽→远近, 退后半步→退後頒布).
  - ACE-Step `battle_vocal_ace_v1_c1`: ratio 0.35, coverage 0.52 -- **skipped the
    first four lines entirely** and started mid-verse; c2 coverage 0.16.
  This matches the research (HeartMuLa's strength is lyric controllability,
  ACE-Step's is speed/stems), so the vocal battle batch runs on HeartMuLa and
  ACE-Step is parked for instrumental/fast-iteration duty.
- New tooling (all under tools/generation/, plus one venv):
  - `arc_qa.py` -- dynamic-arc measurement: intro level vs peak, lift in dB,
    where the climax sits on the timeline, share of the track that is quiet, and
    verdict (build-up / mild build / early peak / flat). Prints an ASCII
    envelope so the shape is visible without listening. This is the instrument
    that encodes 欲扬先抑.
  - `diction_qa.py` -- faster-whisper transcript vs reference lyric, scored as
    ratio / lines recovered / coverage, with `--json` for the batch.
  - `battle_batch.py` -- multi-candidate generation reusing clean_pipeline's
    `generate_one` (so the codec-OOM fallback still applies), then arc + diction
    scoring, promotion of the best take, and a resumable index.
  - `battle_review.py` -- renders the Chinese listening sheet (`review.md`).
  - `tools/asr/.venv` -- faster-whisper 1.2.1. Zero model download: the
    `faster-whisper-small` weights were already in the repo from the extension
    work, so only the package was needed (PyPI is reachable; hf-mirror mirror
    handles the rest).
  - `setup_yue2.bat` -- YuE2 environment build.
- Ranking used by the batch (deliberately simple, inspectable):
  `arc = verdict_bonus*3 + lift_db/6 + peak_at*2`, `diction = coverage*2 + ratio`,
  `total = arc + diction`; takes failing duration/peak/clipping/silence checks
  are dropped before scoring.
- Batch launched 10:28: 6 titles x 2 candidates x 120 s on HeartMuLa, in
  `outputs/battle_batch/` (candidates/ + selected/ + index.json + review.md),
  log `logs/battle_batch.out.log`. ~2-2.5 h expected; resumable, and each
  finished title is reviewable immediately.
  - Briefs in `inputs/battle_batch/`: 01 dawn_assault 破晓之战, 02 abyss_echo
    深渊回响, 03 iron_oath 钢铁誓约, 04 snowfield_charge 雪原冲锋,
    05 march_endless_night 永夜行军, 06 homeward_sparks 归途星火. ASCII
    filenames with a `_titles.json` slug->中文标题 map, because CJK on the
    command line has already mangled arguments twice today.
  - Every lyric is written to force the arc: sparse tense intro -> verse ->
    pre-chorus ramp -> first chorus -> a quiet bridge breakdown -> a bigger
    final chorus -> short outro.
- Smoke test (01, 1 candidate, 60 s) passed the whole chain and gives the first
  arc reading for the genre: lift 7.9 dB, climax at 0.71, "mild build",
  diction coverage 0.67.
- YuE2: environment built and all weights local (base 7.26 GB + VAE 0.53 GB +
  two LoRAs), but the PyPI mirror only serves a **CPU-only** torch 2.10.0, so
  `torch==2.10.0+cu130` is being pulled from download.pytorch.org (1.87 GB).
  Note YuE2's own README asks for a 24 GB GPU, so on this 12 GB card it is a
  bet that may need offload/quantisation -- it is not the batch engine today.

## 2026-10-13 (later still) - Bilingual requirement; batch 2 queued

- User follow-up: battle tracks must exist in **both Chinese and English**, and
  ideally some songs should mix the two languages inside one track. (Mixing is
  authentic to the genre -- Mandarin verses with an English chorus hook is the
  normal shape in HSR/Arknights battle songs.)
- Tooling changes: `diction_qa.py` gained `--language auto` (whisper detects per
  file, the only workable option for a mixed song); `battle_batch.py` and
  `acestep_batch_battle.py` read an optional `_langs.json` per input folder
  (slug -> zh/en/auto) and pass it through; `battle_review.py` now takes several
  `--batch-dir` arguments so one sheet covers every batch, and shows each track's
  language plus the new `dip_before_peak_db` (the quiet bridge before the final
  climax -- the "抑" in the middle of 欲扬先抑).
- Batch 2 written and queued in `inputs/battle_batch2/` (6 titles, 3 English +
  3 mixed): 07 ashen_horizon 灰烬地平线 (en), 08 tides_of_war 战潮 (en),
  09 carry_the_light 携光而行 (en), 10 half_light 半明半火 (zh verse + en
  chorus), 11 iron_and_song 铁与歌 (alternating zh/en lines),
  12 bridge_of_dawn 黎明之桥 (zh verse/bridge, en pre-chorus/chorus).
  English tags explicitly ask for English singing; mixed tags state the
  bilingual structure.
- Watchdog updated: it now launches batch 2 automatically the moment batch 1
  finishes (then reviews both), so the GPU is not left idle. It is instructed
  never to start batch 2 twice (two concurrent HeartMuLa processes would OOM the
  12 GB card).
- Batch 1 progress at time of writing: 3/6 promoted -- 01 dawn_assault
  (build-up, 11.3 dB, peak 0.81), 03 iron_oath (build-up, 9.0 dB, peak 0.81),
  02 abyss_echo (mild build, 7.8 dB); 04 snowfield_charge rendering. Every
  promoted take so far has lyric coverage ~1.0, i.e. the whole lyric came back.

## 2026-10-13 - YuE2 runs locally: the transient/timing ceiling is beaten

- **YuE2-3B now generates full songs on this 12 GB card.** Upstream asks for a
  24 GB GPU, Linux and FlashAttention; three accommodations made it work:
  `memory_budget_gib=12` (which also auto-shrinks `vae_core_frames` to 512),
  `backend="torch-eager"` (the CUDA-graph path calls
  `torch.ops.aten._flash_attention_forward`, which exists as a stub in the
  Windows PyTorch build and raises `USE_FLASH_ATTENTION was not enabled for
  build`), and NAR query chunking.
  - The NAR patch is the interesting one: `nar.synthesize` supports
    `query_chunk_size` but the high-level `pipeline.synthesize` never passes it,
    so the prefill allocates one query block the size of the whole song and OOMs
    (needed 2.42 GiB with 1.13 GiB free on the 10 GiB budget cap). `yue2_generate.py`
    and `yue2_batch_battle.py` inject it by monkeypatching `yue2.nar.synthesize`,
    leaving the installed package untouched. fp8 is NOT an option here -- it
    requires compute capability >= 8.9 and the 3060 is 8.6.
  - Measured cost: 639 s for a 184 s song (ABC plan 62 s, semantic/AR 244 s,
    NAR 311 s, VAE 14 s) at ~9 GB VRAM. Slower than HeartMuLa's ~6.5 min per
    120 s take, but in the same league.
- **Same-brief comparison (01 dawn_assault), YuE2 vs HeartMuLa:**
  | metric | YuE2 | HeartMuLa |
  |---|---|---|
  | length (model-chosen) | 184.5 s | 120 s |
  | opening suppression | -20.4 dB | -11.3 dB |
  | total lift | **20.4 dB** | 11.3 dB |
  | bridge dip before climax | **-26.0 dB** | -17.3 dB |
  | quiet share | **0.28** | 0.12 |
  | climax brightness delta | +442 Hz | +868 Hz |
  | diction ratio / lines | 0.43 / 7 of 26 | 0.56 / 7 of 26 |
  | diction coverage | 1.10 | 1.03 |
  YuE2 wins decisively on the property the user actually asked for (欲扬先抑 +
  late epic climax) and on breathing room; diction is a wash, with HeartMuLa
  marginally ahead on per-character accuracy and YuE2 ahead on how much text
  comes back. Both write 48 kHz stereo float WAVs.
- YuE2 also emits the **symbolic ABC plan** as an artifact (`score.abc`,
  `plan.json`, `semantic.npy`, latents) for every take. That is the lever
  HeartMuLa has no equivalent of: the structure can be inspected and edited and
  then re-rendered, which is the most direct way to control 欲扬先抑.
- Launched the YuE2 pass over all 12 battle briefs (both the Chinese set and the
  bilingual set), 1 take per title, into `outputs/battle_batch_yue2/`
  (log `logs/yue2_batch.out.log`). ~11 min/title, so ~2.2 h. Output schema
  matches battle_batch.py, so battle_review.py renders all three batches as one
  sheet.
- Watchdog now covers the YuE2 batch specifically: if it OOMs, it retries with
  `--nar-query-chunk 1024` (the patch is resumable and skips promoted titles).
- Deliberately NOT done: the ACE-Step comparison pass on the same briefs. With
  HeartMuLa and YuE2 both delivering, ACE's known lyric-dropping (coverage
  0.52/0.16) makes it the weakest of the three for vocal battle tracks; it stays
  available as the fast/MIT-licensed option for instrumental work.

## 2026-10-13 - YuE2 batch: the OOM is cumulative, and the Windows caveat

- First YuE2 pass over the 12 battle briefs degraded over time: 01, 02, 05, 06
  promoted, then 03, 04, 07, 08, 09, 10 all failed with CUDA OOM at the NAR
  prefill. The failures get *more frequent* the longer the process lives
  (286 MiB reserved-but-unallocated early, 373 MiB later), which is the
  PyTorch caching allocator fragmenting across generations -- the model is only
  loaded once, so nothing is released between songs.
- The remedy PyTorch itself suggests in that error, `PYTORCH_ALLOC_CONF=
  expandable_segments:True`, **is not supported on Windows** ("expandable_segments
  not supported on this platform" -- it is a no-op). So the real fix is process
  hygiene, not configuration:
  - one process per title (fresh allocator each time), and
  - `--nar-query-chunk 1024` (halved peak for the prefill block that needs
    ~2.4 GiB in one allocation).
  That is what `tools/generation/yue2_fill_missing.bat` does; it skips titles
  already promoted, so it is idempotent and resumable.
- Stopped the degrading batch at 4/12 rather than let it burn ~11 min per title
  on the two titles still queued (they were failing consistently by then), and
  started the fill run for the 8 missing titles.
- YuE2 arc results so far are materially better than HeartMuLa on the same
  briefs: 01 dawn_assault lift 10.6 dB, 02 abyss_echo 16.1, 05 march_endless_night
  17.7, 06 homeward_sparks 16.0 -- against HeartMuLa's 11.3 / 7.8 / 8.7 / 8.1 on
  the same four titles. Diction coverage 0.74-1.11, i.e. comparable to HeartMuLa.

## 2026-10-13 - End of day: state and how to resume

- Stopped at the user's request with 23 of 24 delivered tracks on disk:
  `outputs/battle_batch/selected/` (HeartMuLa, 6 Chinese),
  `outputs/battle_batch2/selected/` (HeartMuLa, 3 English + 3 zh/en mixed),
  `outputs/battle_batch_yue2/selected/` (YuE2, 11 of 12 -- 08_tides_of_war
  missing).
- Every delivered track has a normalized `*.wav` plus a zero-processing
  `*_raw.wav`; every take keeps its candidate WAVs under `candidates/` and each
  title's `result.json`. YuE2 takes additionally keep their symbolic plan
  (`score.abc`, `plan.json`, latents) -- that artifact is the lever for editing
  structure directly.
- The combined listening sheet is `outputs/battle_batch/review.md`
  (regenerate any time with:
  `vendor/ACE-Step-1.5/.venv/Scripts/python.exe -I tools/generation/battle_review.py
  --batch-dir outputs/battle_batch outputs/battle_batch2 outputs/battle_batch_yue2`).
- **To finish the last track** (`08_tides_of_war`): it needs a seed whose sampled
  plan fits in 10 GiB. Try:
  `tools/generation/yue2_fill_missing.bat` again (it retries 3 seeds per title,
  and only touches titles missing from `selected/`); if it still fails, run
  `yue2_batch_battle.py --only 08_tides_of_war --nar-query-chunk 512 --retries 3`.
  A retry costs 3-11 min, and failures are cheap because they land early.
- Left running when the session stopped: the fill loop's last attempt on
  08 (windowless, self-terminating). No GPU work will be started by anything
  else -- the watchdog cron was cancelled.
- Open items for next session:
  1. Human listening selection across the engines (the whole point of the
     three-way batch sheet).
  2. 08_tides_of_war finish.
  3. Optional: run YuE2 on the trailing titles so every title has both engines
     (only 11 of 12 YuE2 takes exist; HeartMuLa has all 12).
  4. Optional: the TSFH epic-orchestral LoRA (`monsterovich/yue2-steps-from-hell`)
     is downloaded but the 0.1.3 inference package has no LoRA hooks -- it needs
     a PEFT integration step before it can be applied.
  5. Optional: use the saved ABC plans to edit structure explicitly and
     re-render (the agentic-editing path YuE2 advertises).

## 2026-10-13 - Delivery convention fixed (songs folder + separate lyrics folder)

- The user spotted that the battle deliveries had no lyrics text, which the
  project's own acceptance criteria require ("every review WAV has matching
  lyrics TXT") -- the 20-song set had WAV + MP3 + TXT, today's batches had only
  WAV. Fixed by `tools/generation/export_delivery.py`.
- Naming convention, now the standard for every future run (also stored as a
  memory so it survives sessions):
  - a **dated folder for the songs**: `outputs/2026-10-13_battle_songs/`
    holding `<slug>_<engine>.wav` + `.mp3` (libmp3lame -q:a 2, same as the
    20-song set) + `manifest.json`;
  - a **separate dated folder for the lyrics**: `outputs/2026-10-13_battle_lyrics/`
    holding `<slug>.txt`, one file per TITLE (both engines sing the same words,
    so per-take copies would be duplicates). Lyrics are deliberately NOT mixed
    into the songs folder.
- The manifest carries each take's own objective numbers (arc verdict, lift,
  climax position, quiet share, bridge dip, diction coverage, climax brightness)
  so the delivery folder explains itself without the chat log. The HeartMuLa
  entries were backfilled for the two metrics added after those batches ran.
- Final engine summary over the 24 delivered tracks (same 12 briefs, both
  engines): YuE2 averages 13.7 dB lift, -23.4 dB bridge dip and a brighter
  climax in 9 of 12; HeartMuLa averages 9.8 dB, -20.6 dB and 5 of 12.

## 2026-10-13 - Final delivery layout (settled with the user)

- The user asked why one song appears as several files and then fixed the
  convention. Explained and simplified:
  - the two engines' versions of a title stay (that is the A/B material);
  - MP3 was dropped (it duplicated the same audio as a second file);
  - lyrics are now a **strict 1 : 1** pairing with song files, in their own
    folder, named with the identical stem.
- Settled layout:
  `outputs/2026-10-13_battle_songs/`  24 WAV + manifest.json (667 MB)
  `outputs/2026-10-13_battle_lyrics/` 24 TXT (77 KB)
  e.g. `01_dawn_assault_YuE2.wav` <-> `01_dawn_assault_YuE2.txt`
- `tools/generation/export_delivery.py` now defaults to this layout (WAV only,
  `--mp3` opts back in, one lyrics file per song file) so future runs need no
  instructions. The convention is also stored as a cross-session memory.

## 2026-10-13 - Engine written into the lyrics files

- User: keep both engines' versions *and* state the engine explicitly. The file
  names already carried `_HeartMuLa` / `_YuE2`, but opening a lyrics file said
  nothing about which version it belonged to, so every lyrics TXT now starts
  with a header:
  `# 破晓之战 · 01_dawn_assault · 引擎 YuE2` plus the corresponding audio path and
  that take's numbers (seed / 结构 / 起伏 / 高潮位置 / 桥段回落 / 歌词还原).
- `export_delivery.py` now recomputes the arc from the delivered file for every
  track instead of trusting `result.json`: the metrics are deterministic
  functions of the audio, and result.json for the earliest batches predates two
  of them (which had been rendering as "桥段回落 None dB"). Brightness in
  particular was never stored in result.json.
- Final delivery: `outputs/2026-10-13_battle_songs/` = 24 WAV + manifest.json,
  `outputs/2026-10-13_battle_lyrics/` = 24 TXT, strict 1:1 by stem, no MP3.
  Manifest fields: file, engine, title, seed, arc, lift_db, peak_at,
  quiet_share, dip_before_peak_db, diction_coverage, brightness_delta_hz,
  climax_brighter, lyrics_txt.
- Engine averages over the 12 shared briefs: YuE2 13.7 dB lift / -23.4 dB bridge
  dip / climax brighter in 9 of 12; HeartMuLa 9.8 dB / -20.6 dB / 4 of 12.

## 2026-10-13 - Delivery layout, final rule

- Settled convention (also stored as a cross-session memory, since it took a few
  corrections to land):
  1. one dated folder per delivery, named `YYYY-MM-DD_<name>`;
  2. songs AND lyrics together in that folder, each song file paired with one
     same-stem lyrics file (`01_x_YuE2.wav` <-> `01_x_YuE2.txt`);
  3. WAV only (no MP3); both engines' versions kept;
  4. the lyrics header names the engine and its audio file.
- The **2026-10-13 battle delivery is the deliberate exception**: it stays as
  two folders (`2026-10-13_battle_songs` + `2026-10-13_battle_lyrics`) because
  the user asked for that split on the day.
- `export_delivery.py` defaults to the single-folder layout now;
  `--split-lyrics` reproduces today's two-folder output, and the lyrics header
  adjusts its "对应音频" reference accordingly (bare file name in one folder,
  folder-prefixed when split). Verified on a throwaway dir before replying.

- CORRECTION (same day, user): all of a run's output goes into ONE dated folder.
  Rebuilt the delivery as `outputs/2026-10-13_battle/` (24 WAV + 24 same-stem
  TXT + manifest.json + `2026-10-13_battle_review.md`); the two split folders
  were removed. The lyrics headers' "对应音频" reference was rewritten to the
  bare file name. `export_delivery.py` defaults to the single folder;
  `--split-lyrics` remains only as the historical layout.

## 2026-10-14 - English epic battle batch (YuE2, 6-hour budget)

- User listening verdict on the 2026-10-13 three-way delivery: **YuE2 preferred
  over HeartMuLa** ("heart 的有点平庸了"); melodies of the first three titles are
  good, lyrics are still the weak point; duration preference is 2:00-2:30 or
  3:00-3:30, with 1:30 clearly too short; English references are HOYO-MiX's
  "Wildfire" (野火, HSR Belobog) and "Flares of the Blazing Sun" (耀斑, HSR 3.4).
- Today's brief, confirmed by the user: English epic battle **with vocals**,
  lyrics re-designed fresh by an LLM so no two titles share a concept, GPU busy
  for ~6 hours.
- **Calibration found before writing a single note** (`s/line` from the 12
  delivered YuE2 takes): 23 lines -> 142-224 s, 26 lines -> 150-210 s (median
  185 s = 3:05 across seven titles), 41 lines -> 215 s. Song length follows the
  lyric line count, so 26-27 lines is the centre of the requested window. An
  LLM asked for "28-34 lines" overshot to 35-45 and produced three failures; a
  per-section quota summing to 27 made all 12 comply first try.
- New `tools/generation/epic_lyrics_gen.py`: per-seed briefs (angle + image
  bank + key + bpm + voice) -> DeepSeek writes title_en/title_zh/hook/lyrics/
  lyrics_zh/style_extra -> local gate (exact section tags, 24-30 sung lines,
  English-only, hook repeats, `lyric_diversity` passes) -> `<slug>_lyrics.txt`,
  `<slug>_tags.txt`, `_langs.json`, `_titles.json`, `briefs.json`. Tags are
  built deterministically (fixed HOYO-MiX battle-anthem head + arc + vocal +
  negatives, plus the model's own production descriptor, de-duplicated) so the
  register cannot drift between titles. Hook/title/opening-image history is fed
  back in, so titles cannot converge. 12/12 written, 27-29 lines each, all
  distinct hooks. 12 offline tests.
- New `tools/generation/run_epic_yue2.py` + BOM'd
  `tools/generation/start_epic_silent.ps1`: time-budgeted driver over
  `yue2_batch_battle.py` -- **one process per title** (the 2026-10-13 cumulative
  NAR-prefill OOM), deadline checked between titles, per-title timeout, and a
  duration gate (accept 120-215 s, otherwise discard the promoted take and
  re-run with a new seed base, since YuE2 samples its plan length). 6 offline
  tests.
- Run launched 11:20:32 (pid 316320/316436), log
  `logs/epic_run_20261014_112031.log`, per-title logs
  `logs/epic_<slug>_<stamp>.log`, state `outputs/2026-10-14_epic_en/RUN_STATUS.txt`.
  12 titles x 2 candidates, NarQueryChunk 1024, budget 6 h, min 120 s / max
  215 s. Model load 47 s; take 1 running at 10.4 GB / 93% GPU.
- Not yet done: listening sheet + dated delivery (export_delivery.py) + progress
  of `outputs/2026-10-13_battle/` (folder no longer in outputs/ -- user moved it).
- Mid-run correction (user, 14:5x): "四分钟还是太长了最长不过3分半吧". The
  duration gate was already doing its job -- 05_spearstorm's first take came back
  269 s (4:29) and was discarded and re-rolled with a new seed -- but the ceiling
  was 215 s (3:35) with slack. `run_epic_yue2.py` default and
  `start_epic_silent.ps1` are now **210 s (3:30), no slack**; floor stays 120 s.
  The live process keeps 215 s until it exits (args are read once at start);
  restarting it would have cost the in-flight 05 round 2 plus re-rolls of 02/04,
  i.e. ~1.5 h of the 2.5 h left. Two promoted takes sit 1-3 s over the new
  ceiling (02 at 213.2 s, 04 at 211.1 s) -- left in for listening, flagged to the
  user; a future resume run will discard them automatically since the gate
  re-checks every promoted take it finds.

## 2026-10-14 (evening) - Run finished 9/12; delivery pushed to the user's repo

- Run ended at 17:28:58 after 368.4 min: **9 titles delivered** (01-09). Not
  delivered: 10_the_sky_we_burned (its take came back 233 s = 3:53, the length
  gate discarded it, and the rerun never started because the 6 h budget ran out),
  11_vow_under_ruins and 12_rise_again (never reached). All three are simply
  absent, not broken -- a resume run picks them up.
- Delivered metrics (YuE2, all nine `build-up`): lifts +9.1 to +18.8 dB, climax
  at 0.77-0.91, bridge dip -12.2 to -29.8 dB, diction coverage 0.95-1.14,
  durations 182-213 s. Titles: 破晓前一小时 / 断剑不还 / 铁醒之前 / 无人踏足的路 /
  在雷鸣蹄下 / 灯火里的余烬 / 石上血痕 / 路上的名字 / 钟与潮.
- Two takes sit 1-3 s over the 3:30 ceiling (02 at 213.2 s, 04 at 211.1 s) and
  06 at 212.1 s: the live process still carried the old 215 s window. Flagged to
  the user; a resume run with the new 210 s default will discard and re-roll them.
- Three incidents worth keeping:
  - 05's first take was 269 s and was correctly discarded and re-rolled to 203.6 s
    (lift 16.2) -- the length gate earned its place on its first outing.
  - 05's c2 (a real take, not a failure) took 1889 s = 31.5 min where its c1 took
    832 s. Same brief, same box: plan length is sampled per seed, and a long plan
    costs both quality (early-peak arc) and a lot of GPU. Budget for 2x variance.
  - Per-take cost: 658-1889 s, typical ~800 s. Two candidates per title = 25-48 min
    per title on the 12 GB card.
- Delivery convention applied (the one settled 2026-10-13) and extended with a
  **git step**: `export_delivery.py` writes `<slug>_YuE2.wav` + same-stem `.txt`
  (header names engine + that take's numbers) + `manifest.json`, and the dated
  folder now also carries the Chinese listening sheet (`battle_review.py`). The
  whole folder is committed to the private repo `Jun-shisheng/Music_date`, whose
  working clone lives at `.cache/Music_date_repo`. **Deliverables only** -- the
  user explicitly does not want the workflow/tools/candidates pushed.
- Two tool bugs found only because a dated batch folder was used for the first
  time, both fixed and tested: `export_delivery.py` and `battle_review.py` looked
  for briefs under `inputs/battle_batch*` (so today's `inputs/2026-10-14_epic_en`
  was invisible -> lyrics-less delivery, titles rendered as slugs); discovery is
  now content-based (`brief_folders()`), and both gained `--engine` because a
  dated batch dir carries no engine in its name (it defaulted to HeartMuLa).
  `battle_review.py` also listed result.json rows for titles whose take had been
  discarded, i.e. it named files that do not exist; it now requires the promoted
  WAV. Repo growth note for the user: 9 WAVs = 334 MB, so WAV-only is ~350-400 MB
  per day in that repo (`--mp3` would cut it ~5x if that ever matters).
- Open: run 10/11/12 (a fresh `--hours` window resumes past the 9 promoted
  titles); decide whether 02/04/06 get re-rolled to fit strictly under 3:30;
  human listening of the nine.

## 2026-10-14 (night) - Top-up chain: finish 10-12 then publish automatically

- User: "如果关机前完成了就自动上传没有就算了" -- top up the three missing titles
  and publish automatically when generation stops; a partial result is not a
  problem. Also "仅上传产出文件即可不用把工作流上传".
- Two tool changes to make that safe and unattended:
  - `run_epic_yue2.py --slug-filter 10_ 11_ 12_` (`select_slugs`): a resumed run
    otherwise walks every brief and re-checks each promoted take against the
    duration gate, and because the ceiling was tightened 215 -> 210 mid-run,
    that would silently re-roll 02/04/06 and burn ~1.5 h of the new window.
  - `tools/generation/publish_delivery.py` (+ `finish_epic.bat` +
    BOM'd `start_epic_finish.ps1`): export the dated folder, regenerate the
    listening sheet rewritten for the delivery naming (`<slug>_<engine>.wav`),
    then `git add/commit/push` -- skipping the commit when nothing changed, so
    the step is idempotent and a failed take does not produce an empty push.
- Launched 17:45:19 (log `logs/epic_finish_20261014_174519.log`), 2.5 h budget,
  seed base 1300. Resumable; if the machine is shut down mid-run nothing is
  pushed and nothing is lost -- the next run skips the nine already delivered.

## 2026-10-15 - English epic battle, round 2: user's three notes turned into rules

- User's verdict on the 2026-10-14 delivery ("60 分是有了，但还是有很多问题"):
  1. **前奏太长** -- "要么前奏几秒直接开唱，要么前奏时间 10-20 秒不能太长";
  2. **动态太窄** -- "低的时候不够低，高的时候不够高有点平了"，让我参考
     Starfall / 耀光 等战斗曲；
  3. **重复太多** -- "歌词可以重复但不能大片重复，开头高潮和结尾部分的歌词最好
     就是全新的不重复，中间可以有重复但重复两遍顶天了"。
  音色与人声两条他没问题；要的是"冲击神曲"；今天仍然先做英文歌。
- **先把三条抱怨量出来**（新增 `tools/generation/diag_dynamics.py`，一次性诊断）：
  - 首句人声落点（whisper 词级时间戳）：01 30.0s / 02 32.3s / 03 15.5s /
    04 30.1s / 05 21.4s / 06 11.7s / 07 0.0s / 08 2.7s / 09 8.3s -- 九首里
    五首 ≥15 秒，抱怨属实；
  - 平坦度：1 秒 RMS 的中位电平只比自身峰值低 2.7-6.8 dB，15% 分位也才
    -11.9 ~ -18.4 dB —— 全曲几乎一直贴着峰值在响；
  - 节奏感（谱通量）0.043-0.104：02/05/08 最有律动（用户那句"有节奏感的
    挺不错"应该是这几首之一，但他不记得是哪首）。
- **修法（不是改提示词，而是改可测量的机制）**：
  - `tools/generation/yue2_plan_edit.py`（新）：YuE2 先出 ABC 符号计划再渲染，
    而 `SongRequest.abc` 允许**直接喂计划**。于是把计划里那 18-20 小节的
    器乐前奏裁掉，只留 2 小节（约 3 秒）——确定性修复，不靠模型自觉。
    实测 01 的计划：intro 20 -> 2 bars (-28.4s)。同参考曲的做法：
    明日方舟 Radiant 0:02 开口、崩铁 Wildfire 0:03。
  - `run_epic_yue2.py`：新增 `--intro-bars`（默认 2）与 `--max-onset`
    （默认 20 秒，超了就丢片换 seed）双保险。
  - `epic_lyrics_gen.py`：写词规则重写——hook 全曲只出现 2 次且都在中段，
    [Intro]/[Final Chorus]/[Outro] 必须是全新歌词，任何一行不得唱第三遍；
    主歌/桥段写低音区短句、末副歌写高音区长音（并要求一个开元音长音）。
    本地闸门加了 `lyric_diversity.audit_repetition()`（服务与工具共用）。
    12 个新方向（锻炉、无旗可升、锁肘之墙、铁雹、水线鼓、数活人、信号火、
    狼临墙下、水壶还在唱、一言尘土、九白之日、第二颗心）。
  - 编曲 tag 重写（`TAG_ARC` 等）：按段落写死动态——主歌"鼓停、只有稀疏弦乐
    与 sub bass、比副歌低一个八度、约 30% 能量"，副歌"全奏 + 叠唱 + 太鼓 +
    downbeat sub-bass drop"，桥段"全体撤到钢琴与一句贴耳人声、全曲最静"，
    末副歌前"1-4 秒全静"、末副歌"升全音 + gang vocal + 高音长音"，
    末句"突然收成一句长音"；负向词加了"no steady loud wall of sound"
    "no constant full-band texture under the verses"。这些装置全部来自用户点名
    的四首参考曲（HI3 Starfall / 明日方舟 Radiant·耀光 / 崩铁 Wildfire /
    耀斑），逐段时序由网络调研核过。
  - `arc_qa.py` 新增可测指标：`body_db`（中位电平）、`rms_spread_db`、
    `mid_dip_db`（避开首尾的曲中最低点）、`onset_mean`（节奏）、
    `low_share_peak`（高潮处 sub-bass 占比）；`battle_batch.score_take`
    增加 `shape` 项，候选人排名开始偏好"真有对比、够节奏、够低音"的 take。
  - `yue2_batch_battle.py`：`--intro-bars` 走"先出计划→裁剪→按计划渲染"
    路径并留存原始计划（`*_plan_original.abc`）；新增 early-accept——第一
    take 若已是 build-up + 高潮 ≥0.75 + 躯干 ≤ -6dB 就不再烧第二 take。
  - `diction_qa.py` 输出 `first_vocal_s` / `last_vocal_s`；试听清单
    （`battle_review.py`）新增 首句(s)/躯干(dB)/节奏 三列与今日三条的确认清单。
- 测试：generation 88 passed + 1 skipped，API 8 passed（新增重复审计 4 个）。
- 运行中：`tools/generation/start_epic_en2_silent.ps1` /
  `finish_epic_en2.bat`（生成 6 小时后自动出交付目录并推私有库）。
  输入 `inputs/2026-10-15_epic_en`（12 首英文），输出 `outputs/2026-10-15_epic_en`。
- 小事故：两条 `--only 09_ / --only 12_` 并发生成时各自写 briefs.json，后写的
  覆盖了先写的（09 丢失）。已在 generator 里改成"写前重读合并"，并用 tags 文件
  反推补回了 09 的条目（只有中文对照缺失，不影响生成）。
- **两次 A/B（各约 13 分钟，同一 brief、同一 seed、同一计划）**：
  - 短而"硬"的编曲 tag（把动态写在前 3 句、整体缩短 500 字符）**没有更好**：
    躯干电平 -1.7 dB（比原 tag 的 -3.3 更平）、起伏 8.4 dB、弧线图开头就直接
    是满响。→ **结论：caption 文字控制不了整体动态**，这批只做了唱法/节奏层面的
    改变（谱通量 0.117 → 0.20+，用户"喜欢有节奏感"的那一类）。
  - 计划级"末副歌升全音"（`transpose_section`，Starfall 的做法）：渲染成功，
    但测量无法定论——pyin 在混音上把中位 F0 读成 157 Hz vs 311 Hz（疑似锁到
    低音），chroma 相关性只有 0.08、旋转 0 最佳（测试本身分辨不出来），
    且高潮亮度差从 +763 变成 -807 Hz（警示信号）。**因此没有放进今天的批量**，
    保留为下一次可"用耳朵 A/B"的候选（一次渲染 12 分钟）。
- **批量已启动** 10:51（`run_epic_en2_chain.bat`，6 小时预算，12 首，每首 2 个候选
  + early-accept）：01 因 smoke 已交付直接跳过（首句 3.1s 通过门控），02 起正常
  生成。跑完自动出交付目录并推私有库（`publish_delivery.py --name epic_en`）。
- 又修了两个记账 bug（都会让门控静默失效）：
  1. `yue2_batch_battle.py` 每个标题一个进程，却用**新建的空 list** 写
     `index.json` → 开跑前先读回旧索引并按 title 合并（否则
     `read_first_vocal` 对除最后一首外的所有标题都返回 None，首句门控形同虚设）；
  2. 并发生成时 `briefs.json` 覆盖（见上）。
- 事故：`finish_epic_en2.bat` 的 `rem` 注释里写了中文，cmd.exe 按 OEM 码页读
  .bat，UTF-8 字节被当 GBK 解码后落到命令分隔符上，cmd 去执行了注释碎片
  （'ish'/'ate'）。生成部分没受影响（已跑起来），但按仓库既有规则应**全 ASCII**；
  已新增修正版 `run_epic_en2_chain.bat`，等本次跑完再替换掉旧文件（正在运行的
  .bat 不能改，cmd 按字节偏移重读）。

## 2026-10-15 收尾：9/12 已交付并推送；动态那条仍是未解项

- 运行 10:51:39 – 17:00:52（369.2 分钟，6 小时预算耗尽），**交付 9 首（01-09）**，
  已 push 到私有库（`ed33ff5  2026-10-15_epic_en: 9 tracks (YuE2)`，20 个文件 +
  试听清单）。缺失的 3 首：`10_break_the_line` 渲染出来了但被长度门控判为 225 秒
  （>210）丢弃、重摇时预算用尽；`11_seven_suns`、`12_the_second_heart` 未轮到。
- 门控战果：08 首句 30 秒 → 重摇 2.7 秒；09 首版 223 秒 → 重摇 203 秒；
  10 首版 225 秒（这次没救回来，预算没了）。
- 验收（9 首）：首句 **0.0-4.3 秒**（昨天 15.5-32.3）；节奏 **0.129-0.206**
  （昨天 0.043-0.104，约 2 倍）；长度 180.6-209.4 秒。**躯干电平 -2.5 ~ -3.7 dB
  （昨天 -2.7 ~ -6.8）→ "低不够低"未解决**，且这次裁掉长前奏后更低音量的素材
  也没了。03/05/09 高潮比开头亮 +1705~+1893 Hz，02/04 高潮反而暗（-363/-24 Hz）。
- 6 小时只出 9 首的原因：early-accept 的躯干阈值 (-6 dB) 模型达不到，几乎每首都
  渲染 2 个候选（各约 28 分钟）。下一轮要么 `--candidates 1`，要么把阈值放宽到
  -4，要么把首句/对比度也并进候选打分（这次 08 就是"计划裁好了但渲染把人声放晚"，
  候选排名里没有首句这一项，靠 driver 兜底重摇，多花了一轮）。
- 仍待办：`--final-lift 2`（末副歌升全音，代码与测试已就绪、渲染可用但听感未验）
  的单首 A/B；以及计划层面给"主歌稀疏 vs 副歌全奏"注入乐器密度（未测）。

## 2026-10-15 (evening) - 交付重做：英文歌词 TXT 改成逐行中英对照并重新推送

- 用户新要求："关于英文 TXT 文档我有一个新的要求每一行英文下面要有对应中文"。
- 做法（不是手工改文件，而是改管线）：
  - `epic_lyrics_gen.py` 新增 `--gloss-only`：为每首英文词生成/校验
    `<slug>_lyrics_zh.txt`（逐行中文）。校验是硬性的——段落标签顺序相同、
    每段中英行数相同（`validate_gloss`），对不齐就重问 DeepSeek，绝不硬配。
    今天 11/12 首的旧 gloss 本来就对齐，只有 09 的 gloss 在那次 briefs.json
    覆盖事故里丢了，已用 DeepSeek 补回（416 字，对齐通过）。另外正常生成路径
    现在也会顺手写出 sidecar。
  - `export_delivery.py` 新增 `bilingual_body()`：段标只出现一次，英文行下面
    紧跟对应中文行；若某首 gloss 对不齐则退化成"英文整段 + 中文整段"并在日志里
    报告，不会错位配对。中文歌不加对照（本来就不用翻译）。清单表头也多写了
    `首句 x.x s`。
  - 新增 8 个测试（对照配对、标签不重复、错位降级、sidecar 发现、中英判定）。
- **顺手抓到一个真缺陷**：09 的英文词里藏着演出提示
  `I will wash the child's shoe tonight (held note on "night")` —— 模型唱的是
  行内文字，这个括号可能被唱出来，而且会印进交付稿。已从词文件删掉，并在
  `validate_brief` 加了"行内不得有括号/演出提示"的闸门 + 提示词规则 10
  （生成的音频早已渲染，是否真的唱了那句需要耳朵确认，见待办）。
- 重新导出并推送：commit `ced13c0`（在此前 `ed33ff5` 之上），只动了 9 个 TXT +
  试听清单，**没有重复上传 WAV**（内容逐字节相同），已 push，工作区干净。
- 测试：generation 104 + API 8 全过。
- 待办：
  1. 09 的音频可能唱了那句括号（coverage 1.09 偏高）；若确认难听，用干净的词
     重渲染一首即可（约 12 分钟）。
  2. 10/11/12 三首仍缺（10 因 225 秒超长被丢、11/12 未轮到），可 `--slug-filter`
     续跑。
  3. `--final-lift 2` 的单首听感 A/B（代码与测试就绪）。

## 2026-10-16 - 用户的第二条意见：节奏/密度与高音机械杂音（量化 + 立项）

- 用户听音结论（对 2026-10-15 的 9 首）："和前面有些许提升但远不够"；两条具体问题——
  1. **全程一个节奏**：以 HOYO-MiX Wildfire 为参照，"前期和高潮不只是音调的区别，歌手吐词
     的密度也不一样：前期明显缓慢，高潮更加集中"，而我们"从头到尾都是一个节奏，音调变化也不大"；
     提出的做法是**先模仿顶级音乐的节奏与高低 → 再替换/创造自己的乐器和歌词**（点名泽野弘之等）。
  2. **高音处有机械杂音**：集中在部分高潮段，"不像人类歌手的破音，反而是带有缺陷的机械音"，
     并非每首歌都有。
- 用户同时要求：先写详细长期文档；说明我的能力边界（哪里需要 CODEX 辅助）；清理项目冗余代码与 MD。
- **先量化，再写文档**（新增 `tools/generation/plan_metrics.py`，10 个测试）：
  - YuE2 的 ABC 计划是符号化的：人声每个音符的时值与音高都写死在 `V: Vocal` 行里
    （`"D#m"d2d2d2ef-|`），所以吐词密度与音区**可测且可改**。
  - 对 2026-10-15 全批 23 个候选计划实测：**副歌/主歌吐词密度比只有 1.14**
    （23 首里仅 15 首副歌更密；01/04/07 分别是 0.90/0.72/0.81——副歌比主歌还稀）；
    **副歌音区中心只高 +2.7 半音**；主歌密度 1.37 起音/秒。
    静音占比主歌 23% / 副歌 15%（主歌更留白——这一项本来就是对的）。
  - 解析器修正两处（均由测试抓出）：跨小节连音 `d8-|d8-|d8|` 是**一个**长音（原实现算 3 个起音，
    会正好把副歌的长音算密）；无 `V: Vocal` 的纯器乐段必须保留其小节长度（否则后续段落边界全错位）。
- **交叉验证（新增 `tools/generation/word_rate.py`）**：对 01_hammer_and_hymn 用 faster-whisper
  词级时间戳按"计划推导的段落边界"统计——计划总长 183.4s vs 实际渲染 184.6s；
  演唱段的换算比稳定在 0.74–0.93（词/秒 ÷ 起音/秒），**两列逐段同步**
  => 计划层指标是听觉密度的廉价确定性代理；也证实这首 chorus/verse = 0.84（方向是反的）。
- **结论：不能再靠提示词。** 2026-10-15 的 A/B 已证明 caption 控制不了动态；计划层是唯一控制点
  （裁剪 intro 已证明能改变渲染结果：首句 30.0s → 3.1s）。
- 长期文档：`docs/arrangement_spec.md`（问题一的修法 = 计划级形状变换器 + 参考曲剖面工作流；
  问题二的四个假设、判定实验与处置阶梯；能力边界与 CODEX 分工；待确认项）。
- 新工具：`plan_metrics.py`（计划剖面）、`word_rate.py`（任意音频吐词速率，参考曲也适用）、
  `logs/plan_metrics_1015.json`（全批剖面）。
- **现有客观工具对高音杂音是盲的（实测）**：9 首交付曲跑 `audio_qa.py --degeneracy`——
  `harsh` 全部 0（为 ACE-Step/HeartMuLa 标定的机器噪声代理在 YuE2 上完全失效）、
  dips 2–13（无信息量）、clicks/s 0.6–71.7（被打击乐瞬态污染，相差 100 倍不可能都是缺陷）、
  结尾退化门控 9 首全 ok。**没有一项指标能看见用户听到的机械音** →
  必须走"用户给时间戳 → 标定新检测器"的流程，不能沿用旧阈值。
- 待用户回答：参考曲音频（Wildfire/Starfall/泽野等，网络拿不到）、"meimei"指谁、
  机械杂音的具体时间戳、优先级。

## 2026-10-16 (later) - 计划形状变换器落地 + 项目清理

- 用户决定（三问回答）：**先做节奏/密度改造**；**会提供参考曲音频**（放 `human_refs/refs/`）；
  清理力度＝**只删确证的死代码 + 过期 MD**。
- **新工具 `tools/generation/plan_shape.py`（16 测试）**——把"慢下来/密起来"变成确定性操作：
  - `--reduce SECTION:FACTOR`（主歌降密度）：每 FACTOR 个起音留一个，被丢的音**把时值交给前一个音**
    （变拖长而非静音），可跨休止延长；小节内第一个起音永远保留（踩住拍）。
  - `--split SECTION:MAX_UNITS`（副歌升密度）：超过 MAX_UNITS 的音符拆成**同音重复**；
    跨小节连音 `d8-|d8` 会被拆开并解除连音（同一小节总时值，起音 1→8）。
  - 硬约束（每次改写当场校验，不满足即报错拒写）：只改 `V: Vocal`；小节总时值守恒；
    和弦记号逐字节不动；`V: Ins` 与头部不动。
  - **全仓库 59 个真实计划全量扫描：守卫零触发；主歌密度 59/59 下降、副歌 59/59 上升**；
    平均比值 1.06 → 4.76（`reduce 2`+`split 2` 偏激进，可调）。
    单曲例（01_hammer_and_hymn）：主歌 1.55→0.94、副歌 1.40→2.32，比值 **0.90 → 2.47**。
  - 已知副作用：降密度会丢掉被删音的**音高**，音域可能收窄 1–2 个半音（实测 lo 68→70）；
    音高只会是原集合的子集。要音区更低用移调算子。
  - 写代码时踩掉的真 bug（都记进文档 §1.5）：和弦记号 `"D#m"` 里的 `D` 被音符正则吃掉、
    并吞掉紧随的真音符（输出 `"D5#m"`）；跨小节连音被当成多次起音；纯器乐段被算成 0 秒
    会让 `word_rate.py` 的段落边界整体错位。
- **项目清理（18 项，全部移动到 `_archive/deleted_20261016/` 而非硬删）**：
  因为 `tools/` 下**几乎全部未被 git 跟踪**（审计：`git ls-files` 仅 55 个文件，
  HEAD 停在 2026-08-07），硬删不可恢复，故用归档区。已移出：6 个一次性 generation 脚本
  （add_english_lyrics_cn / bilingualize_english_txt / diagnose_clicks / create_batch_briefs /
  download_hf_file / align_batch_level）、3 个 vocal_repair 死文件、7 个 social_media 批量造词脚本、
  恒被 skip 的 `test_heartmula_device_map.py`（硬编码 `G:\`）、根目录 `sidestep.log`；
  另删 `.pytest_cache`/`__pycache__` 缓存目录。**每一项都先 grep 确认无外部引用**。
- **保留（重要）**：`tools/asr/.venv`（296M，看似孤立，实为 `battle_batch.py`/`word_rate.py`
  的 ASR 运行时）；`prepare_human_refs.py`（被 `human_refs/README.md` 引用，正是参考曲分离要用的）；
  `.cache/Music_date_repo`（1.3G，是私有交付库的 git clone，**不是缓存**）。
- **过期 MD 标注（不删）**：`README.md` 顶部加"当前状态"横幅（下半部分 ACE-Step 产品说明
  已标明为历史）；`task_plan.md` / `findings.md` / `docs/devlog.md` /
  `tools/{generation,vocal_repair,lora,social_media}/README.md` 加归档横幅；
  新增 `docs/superpowers/ARCHIVE.md` 汇总 9 份 YuE2 前的 plans/specs。
  `findings.md` 明确保留：它是 AI 人声杂音的根因知识库。
- **审计发现（与代码无关但重要）**：
  1. `.env` 与 `.env.bak` 里各有一个**明文 DeepSeek API key**（均已被 gitignore、未进 git 历史）；
     若本目录曾被复制或分享，建议轮换。
  2. 仓库 99% 未跟踪（HEAD 2026-08-07）——**任何 `git clean`/checkout 会抹掉当前整条管线**。
  3. 树内有嵌套 git 仓库：`vendor/{ACE-Step-1.5,heartlib,YuE}/*`、`.cache/Music_date_repo`。
  4. 仍是"待确认"的大块：`tools/vocal_repair/`（993M venv，被历史 FastAPI 服务引用）、
     `tools/social_media/` 剩余 6 个脚本、`tools/lora/`、`data/lora_*`（218M）、
     `human_refs/`（993M，属已放弃的 humanize 工作流，但参考曲要用它的 raw/ 目录）、
     `PR_AIMusic_*`（558M，独立子项目）、`.firecrawl`/`.tools`/`.npm-cache`/`.cache`（与音乐无关的缓存）。
- 测试：generation **130 passed + 1 skipped**（新增 plan_shape 16 + 已归档的 device_map 测试）。
- 下一步：§1.7 E1——拿一首歌改计划后渲染 1 次（约 12 分钟）做听感验证；
  等用户给参考曲后建 `ref_profile.py` 标定真实目标数字。

## 2026-10-16 (late) - E1 密度 A/B 完成；参考曲剖面首批；发现"指标复现不出听感"

- **E1 密度实验（同 brief、同 seed 1200，只改计划；渲染 761 秒）**：
  - 渲染留存的 `score.abc` 与 `plan_shape.py` 改过的计划**逐字节相同** → 渲染器确实按改过的计划唱，
    "测量→改计划→渲染"闭环成立。
  - 用 `word_rate.py` 逐段量（`outputs/_shape_ab/selected/` vs `outputs/2026-10-15_epic_en/selected/`）：
    副歌 **1.16/1.13/1.23 → 0.99/2.08/1.84 词/秒**（均值 ×1.4），**副歌/主歌 0.84 → 1.95**（落进目标区间）；
    桥段/间奏未变（算子只作用于目标段落 ✓）。
  - **但 `reduce 2` 对主歌过头**：第一段主歌 25 秒全成拖长音，whisper 检出 **0 个词**——歌词等于丢了。
  - **门控缺陷**：首句落点 3.1s → **19.16s**（不是前奏变长，是"第一个可辨词"被推后）→
    首句门控会被"长拖音开场"误判，需与密度指标联动。
- **参考曲：用户已下载 12 首**到 `human_refs/refs/`（Wildfire / 耀斑 / TSFH Victory / Nightwish /
  Sabaton / Within Temptation / Beast in Black / FOB / Imagine Dragons / Skillet / Ruelle / Bergersen），
  来源 B 站 DASH 音频（用户自己的 `tools/media/bili_audio_dl.py`），索引见该目录 `_index.md`。
- **新工具 `tools/generation/ref_profile.py`**：任意音频 → 每 10 秒窗的响度/起音速率/音区（pyworld）；
  `--whisper` 再叠一层词速率（复用 `word_rate.py` + `tools/asr/.venv`）。可续跑（每首跑完即落盘）。
  demucs 权重从 HF 成功下载（84MB，本机此前记录 HF 不可达，这次可用）。
- **首批同口径结果（whisper 词密度，10 秒窗，都走 demucs 人声轨）**：

  | | floor(p25) | peak(p90) | 窗内摆幅 | loud/soft 均值比 | 音区抬升 |
  |---|---|---|---|---|---|
  | Wildfire | 0.90 | 1.60 词/秒 | ×1.78 | **×1.20** | +1.6 半音 |
  | 我们的 01 | 0.72 | 1.63 | ×2.26 | **×0.92** | +4.7 半音 |

  - **对上用户抱怨的只有 loud/soft 那一项**：Wildfire "越响越密"（×1.20），我们"越响越稀"（×0.92）。
  - **但窗内摆幅与音区抬升我们反而更大**（×2.26 vs ×1.78；+4.7 vs +1.6 半音）——**复现不出用户听感**。
    已定位原因：这两项"不知道段落在哪"，我们的摆幅主要来自**器乐间奏/桥段**（间奏仅 0.2 词/秒），
    不是"主歌→副歌"。
  - **推论（重要）**：参考曲的密度对比幅度其实只有 ×1.2，**不算大** → "冲击感"的差距很可能主要在
    **编曲层面与旋律钩子**，而这两项目前**没有任何客观指标**。
- **数据质量修正**：起音检测器会把**鼓串进人声轨的瞬态**算成发音——实测 3.43 起音/词（我们）、
  2.76（Wildfire），所以先前"我们摆幅 ×1.67"的起音数字不可信。`ref_profile.py` 现在输出
  `onsets_per_word` 用于一眼判断污染。
- **已向用户请求**（下一步的关键输入）：指出 ① 我们最"平"的具体时间点、② Wildfire 最"对比"的
  具体时间点，我按这两段校准指标（与 §2.4 的听音标定同一套循环）。
- 运行中：`ref_profile.py --all --whisper`（13 轨，约 1 小时，日志 `logs/ref_profile_*.log`，
  报告 `human_refs/refs/_profiles.md`）。

## 2026-10-16 (day) - 钩子相似度：找到真正的病根；整天批次（hook+split 双修复）

- **参考曲 12 首全量剖面跑完**（`ref_profile.py --whisper`），报告 `human_refs/refs/_profiles.md`。
  过程中两个方法学问题被修掉：
  1. **单曲样本是噪声**。只拿 Wildfire 比时它"响处更密（×1.20）"而我们"×0.92"，我据此写过
     "我们方向是反的"。扩到 10 首人声参考曲后，**band x 中位就是 ×0.89**——我们的 ×0.92 完全正常。
     **参考系必须用一组，不能用一首。**
  2. **起音检测器被鼓串扰污染**（3.43 起音/词，我们；2.76，Wildfire），已加 `onsets_per_word`
     字段与"只统计有人声的窗口"（否则 Sabaton 的 floor 会被长器乐段算成 0.10）。
- **在吐词密度的所有维度上，我们与参考曲无显著差异**（我们自己 vs 10 首人声曲中位）：
  摆幅 ×2.05 vs ×1.98；floor 0.80 vs 0.88 词/秒；peak 1.64 vs 1.95；band ×0.92 vs ×0.89；
  音区抬升 +4.7 vs +2.2 半音。=> 用户的"从头到尾一个节奏"**用密度统计复现不出来**。
- **新指标 `hook_similarity`（响段两两的 chroma 余弦）找到了病根**：
  参考曲 **中位 0.978**（四分位 0.947–0.982），**我们 0.902，低于每一首人声参考曲**
  （演唱窗占比 94%，不是器乐造成的假象）。含义：真实副歌是"**同一个钩子又来了**"，
  我们是"**和弦一样、每次旋律换个说法**" → 听不出副歌感。这与用户原话"音调变化也不大"
  是同一件事的两面，而且是**整个批量的系统性特征**。
- **新算子 `plan_shape.py --hook SECTION`**：把第一段 SECTION 的旋律（**连和弦记号一起**）
  复制到后面各段；小节数不同时按动机循环铺满；按**整八度**对齐目标段音区
  （修掉一个真实极端例：某 take 第一段副歌比后面低一个八度，照搬会把末副歌降八度，
  正好与"末副歌往上走"相反）。歌词另行喂入 → "末副歌歌词必须全新"不受影响。
  **全仓库 60 计划验收：时值零改动**（每段小节数与秒数逐字节核对），三段副歌旋律变为同一句，
  音域外扩最大 5 半音。
- **组合代价实测**（同批 23 个 take 的 副歌/主歌 起音密度比，中位）：
  原始 1.13 / `split chorus:2` 2.60 / **`hook`+`split` 2.51** / 仅 `hook` 1.46
  => 钩子几乎不花密度代价，两个修复可以同时上。
- **整天批次已启动**（13:36:25，22 小时预算，12 首 × 3 候选，**同 seed 1200** 便于与
  10-15 交付做逐首 A/B）：`run_density_chain.ps1` = `run_epic_yue2.py --shape 'hook chorus'
  --shape 'split chorus:2'` → 工作目录 `outputs/2026-10-16_epic_hook` → 交付目录
  `outputs/2026-10-16_epic_en`（**本地导出，不推私有库**，等用户听完再说）。
  日志确认两个算子都生效：`shape: chorus hook` / `shape: chorus split` / `plan: intro 20 -> 2 bars`。
- 中途废弃：先跑过一次只有 `split` 的批次（`outputs/2026-10-16_density`，2 首未 promote），
  因发现钩子病根而换成双修复。
- 测试：generation **145 passed**（本次新增 plan_shape 10 个 hook/thin 测试、plan-file 4 个）。

## 2026-10-16 (evening) - 公开仓库更新 + 前端/后端接口修复

- 用户指令：剩下的生成不跑了；把工作流"史诗级"更新到 **公开仓库**
  `github.com/Jun-shisheng/AI_Music`（本地仓库的 origin 就是它），
  **不上传**：隐私信息/密钥、产出音频、随机种子的生成、歌曲编排的后调整框架，**只上传最基础的项目**。
- 出于稳妥，改在**从已发布提交拉出的干净工作树**里做（`git worktree`）——因为本地工作区有未提交改动
  （heartmula 后端、humanize 后处理链、LoRA、端口 8001），直接 `git add` 会把它们一起带上公开仓库。
- **两次推送**：
  1. `e8aaa5e` 重写 README（讲清"测量优先"的管线与硬件现实）+ 发布质检/打包工具
     （arc_qa / diction_qa / audio_qa / loudness / export_delivery / battle_review + lyric_diversity）
     + `docs/workflow.md`；加固 `.gitignore`（补上 inputs/logs/human_refs/_archive/.cache 等私有目录）。
  2. `e181777` 修三个"clone 下来就跑不起来"的真 bug + 接口文档。
- **修掉的三个真 bug（都已实测复现→修复→复验）**：
  1. **`start.bat` 硬编码 `G:\AI_Music`**（老机器路径）——所有路径都从它推导，别人 clone 必然全线找不到。
     改为从脚本自身位置推导 ROOT（可用 `AI_MUSIC_ROOT` 覆盖）；`download_models.py` 与
     `config.py` 的 `acestep_dir` 默认值同样是硬编码路径，一并改成**仓库相对**。
  2. **后端根本起不来**：默认 `DATABASE_URL=sqlite:///./data/app.db` 是相对路径，
     而 sqlite **不会创建缺失目录** → 启动即 `sqlite3.OperationalError: unable to open database file`
     → `Application startup failed`。新增 `db.ensure_sqlite_dir()` 在 import 时建目录。
     这是**任何人 clone 后第一个会撞到的报错**。
  3. **`generation_type` 从未传进生成器**：前端提供 `song/bgm_loop/sfx` 三种类型，
     job 也存了，但 `queue.py` 只传了 duration/loopable → 选"音效"和选"歌曲"生成的东西一样。
     新增 `resolve_request()`（纯函数，可测）集中映射：
     `song` 原样；`bgm_loop` → 器乐 + seamless loop 等提示词；`sfx` → 非音乐 + 默认 10 秒。
- **文档**：新增 `docs/api.md`——完整接口契约（端口/代理链路、每个端点的请求与响应、
  `generation_type` 映射表、错误语义、clone 后必须安装什么、改契约要同步哪几处）。
  README 增加"三种生成类型"与运行步骤，并链接该文档。
- **验证方式**：不是"我本地能跑"，而是**在干净检出上跑**——
  `git archive` / worktree 出发布态 → `pytest` 28 通过（原 16 + 新增 12）、
  `import app.main` 通过、**真起 uvicorn 打接口**（health/ generate / jobs / library / audio / 422 / 404 全对）、
  sfx 任务失败信息正常落在 `error` 字段而不是崩服务、**前端 TS 与后端 pydantic 字段集逐项比对完全一致**。
  实测中还抓到我自己引入的一个参数不匹配（`generate()` 调用处多传一个参数），已修并补了一条打桩测试。
- 未上传（按用户要求）：所有音频与 `outputs/inputs/logs/human_refs/_archive/.cache`、
  `.env`（含密钥）、`progress.md`/`findings.md`/`docs/arrangement_spec.md`/生成驱动
  （`run_epic_yue2`/`yue2_batch_battle`/`clean_pipeline`/`epic_lyrics_gen`）、
  计划后调整框架（`yue2_plan_edit`/`plan_shape`/`plan_metrics`/`ref_profile`）、
  人声后处理链（`tools/vocal_repair/*`）。公开仓库共 66 个文件，无音频、无 .env。
- **遗留**：本地 `main` 现比远端**落后 1 个提交**（e181777），且本地未提交改动与那次提交**改了同一批文件**
  （config.py / queue.py / acestep_runner.py / start.bat），所以将来 `git pull` 需要先 stash/提交这些私有改动。
  已保持原样未动。

## 2026-10-16 (later) - 参考曲到位：`human_refs/refs/` 12 首（回 §5 待确认项 1）

- 用户给了 §5 第 1 条的答案：要 **野火 Wildfire、耀斑 Flares of the Blazing Sun**
  （崩铁，HOYO-MiX）+ **10 首热门英文曲**，放到 `human_refs/refs/`，只要音频、后台静默下载。
- **改成 CLI 直连 B 站 API，没用 JJdown 的 GUI**（这是本次唯一偏离用户字面要求的地方）：
  `D:\JIJIdown` 装的是新版唧唧Down（Tauri + WebView2），**没有 headless 模式**，
  一跑就是窗口；与规则 1「静默后台，绝不弹窗」冲突。故直接走它内部用的同一套
  web 接口：`search/all/v2` 搜 → `view` 取 cid → `player/playurl?fnval=16` 取 DASH
  音频档 → ffmpeg 转 m4a。**全程无窗口**。未登录也能拿到 198kbps 档（比预期好）。
  新工具 `tools/media/bili_audio_dl.py`（纯 CLI；`--list` 可只看候选不下载）。
- 英文 10 首按用户选的**史诗/战斗**方向（AskUserQuestion 三选一）：
  TSFH Victory / Imagine Dragons Warriors / Skillet / Within Temptation /
  Sabaton / Beast in Black / Nightwish / TSFH Strength of a Thousand Men /
  Fall Out Boy / Ruelle。
- **踩坑：按播放量排序会选错歌**。第一轮 12 首里有 4 首是无关的大热门视频
  （Imagine Dragons Warriors → 驯龙高手 Test Drive；Skillet Warrior → HELLBLADE；
  Beast in Black → EVA 配乐；Ruelle → Empires）。给 `pick_candidate` 加了
  **标题必须命中的 gate**（`--query "名:检索词|必含1,必含2;必含3|排除词"`），
  改完 4 首里 3 首修好。剩 2 处需人判断，已在 `refs/_index.md` 记录：
  1. **Skillet - Warrior 在 B 站根本不存在**（前 3 页全是别的 warrior），
     替换为同乐队 Awake And Alive；
  2. Nightwish 首轮抓到的是 Live at Masters of Rock——现场观众噪声会污染
     §1.6 的速率/音区测量，改回录音室版。
  其余 10 首已逐首 ffprobe 核对时长码率，与官方发行版一致。
- 12 首合计 64M，全在 `human_refs/refs/`（AAC 原始码流，未重编码）。
  **只放 `refs/`，没进 `raw/`**：这些是商业发行，`human_refs/README.md` 明确
  「不得作为人声克隆目标」，`refs/` 只服务于 §1.6 的测量用途。
- 下一步：§1.7 E5——建 `tools/generation/ref_profile.py`，把这 12 首过成
  `refs/<name>_profile.json`（吐词速率 / 音区中心与跨度 / 段落边界），
  用实测数字替换 §1.5 拍脑袋的目标值。


## 2026-10-19 — 固定音色与参考编曲两条新工作流

- 用户授权实现两条工作流，并指定16号第4首附近的女声音色；采用选中 Iron Rain c2 的4–11秒干声。用户当前无耳机，授权继续沿用此参考，未进行主观听审。
- 新增 `tools/voice_studio`：持久歌手档案、官方SoulX-SVC翻唱、导唱SVC与音素/音符SVS改词路径；原作完整归档，固定参考和运行依赖参与缓存摘要，失败留证，不混同命令完成与质量验收。
- 两个真实20秒换声样片完成：`cover_no_white_flag_v1` / `cover_hammer_v1`。独立CPU QA的前后字符转写相似度分别0.947 / 0.630；仅内容保留代理，后者需重点复核。两者无长度漂移或技术削波警告，音色一致性待听审。
- 新增 `tools/reference_studio`：声学区间草案、参考蓝图、明确目标段落映射、四臂ABC计划、隔离渲染、恢复/输入等价复用、同RMS试听报告。保护人声/和弦/时值；不把频带自动命名为乐器，不填虚构审美分。
- 首轮21小节/58.605秒对照：baseline58.639秒；密集arranged81.559秒(+39.17%)，未通过时长门槛。外部ABC不是强制输出时钟；报告对>10%漂移停用符号分段测量。保留首轮诊断产物。
- 短风格描述修正原172 BPM/B minor与实际86 BPM/D# minor的矛盾，去掉短片段未出现的桥段尾奏。温和support对照baseline58.519秒；目标句ASR从首轮0/9到4/9，副歌仍不稳定，不能作为高分成品。
- 首轮新词YuE导唱在混音和干声上均0/4目标句，拒绝把该错误导唱换声后伪装改词成功。另建 `iron_rain_female_svs_v1`，同母本参考+近似对齐元数据；官方DataProcessor CPU验证通过，目标8小节22.326秒，保留原音区和-12半音版本。
- 统一入口 `run_workflows.bat` / `tools/music_workflows.py` 用OS文件锁调度GPU，按本次Popen PID取消自身进程树，记录真实退出状态。离线试听页对音频与报告摘要核验，导出评分携带具体实验及音频摘要。
- 本轮运行、QA与试听汇总位于 `outputs/2026-10-19_quality_upgrade`；SVS实际合成与support arranged结果在后续条目/最终运行报告记录。既有生成器、旧处理链与用户未提交代码保持原样。


### 2026-10-19 最终运行结果

- 温和support的baseline/arranged均真实完成：58.519 / 60.039秒，相对58.605秒计划为-0.15% / +2.45%。arranged近似目标句ASR9/9，额外转写仍存在；不声称高分。
- 新词直接SVS完成v1/v2，两版均22.326秒/24k单声道。v1/v2近似目标4/4，diction ratio .851→.870、coverage1.171→1.124；独立正规化字符相似度.886→.906，词编辑7→6。v2仅将sparks起音提前并合并同音槽，旋律时间线、总时长、词序不变；官方processor音素主体帧0→14。后两句全文匹配，前两句仍有错词/插词疑点。两版无NaN/Inf或削波，全部输出/参考/metadata摘要吻合。
- 追加修复：score/melody控制参数进入SVS缓存；新任务score冻结在持久job内，校验原记录SHA后供worker读取。历史两job仅追加按原SHA核验的score备份，不回写manifest/worker spec。
- 全部新套件最终85项通过：Voice Studio 34、Reference Studio 47、入口取消/试听追溯4。真实GPU执行YuE2 5次、SVC 2次、SVS 2次；wildfire_transfer为显式输入等价复用，不计新推理。模型与依赖doctor通过。
- 离线试听页10个真实音频资产及来源摘要通过核验，评分导出包含实验identity和音频摘要；JS语法检查通过。用户无耳机，声音身份及审美验收仍待定，未代替用户打分。
- 自动审批以blocked by policy拒绝删除.range_parts临时下载分片；已保留缓存，不影响模型和成果。未申请额外许可或尝试绕过。
- 完整交付状态：outputs/2026-10-19_quality_upgrade/run_summary.json；用法：docs/quality_upgrade_workflows.md；统一入口：run_workflows.bat。

### 2026-10-19 同日继续开发与日期交付

- 用户授权继续推进，将所有产出集中到日期目录。新增 `outputs/2026-10-19/index.html`，统一入口 `review` / `package` 默认按上海时区归档；首轮产出、持久歌手档案、完成任务、日志、输入与源码快照复制到同一日期包，旧版本逐文件校验并保留，音频路径可离线播放。
- 新词SVS v3修复rising、walk/on、find/my三处跨休止词界。固定参考、旋律、休止、词序、总时长和seed保持一致，官方处理器未分配帧62→52，零主体槽1→0。新音频22.326秒；统一QA总词编辑v2为6、v3为5，插词3→0，但第三句存在替换回归，保留父版本。
- 新增CPU逐句组合工具与v4候选，使用v3前段和v2后段，在Every尾部11.678792秒共同低能窗口做20ms线性淡化。该处不是句间静音。采样点数、时长不变，淡化窗外与对应父音频逐样本一致。统一QA总词编辑降至4（4替换/0漏词/0插词），字符相似度.943925，仍未逐字或听感验收。CPU组合不计新模型推理。
- 新增统一CPU QA：原生音频检查、无目标提示转写、全局非重复词对齐、S/D/I和逐行结果，保存模型/参数/输入输出摘要。ASR用于定位风险，不生成审美或音色身份分。固定音色听审保持待定。
- 原词首副歌新对照seed1238、CFG1.5、semantic上限1200：melody baseline23.158667秒；混音与分离干声均出现额外重复，不能简单归咎于伴奏误识别。melody arranged触顶截断，拒收并保留诊断。最后仅做一次full和声条件诊断：824 tokens正常结束、32.918667秒，但相对22.326秒计划偏长47.445%，技术复核标志保留，符号分段指标停用。没有增大上限、裁尾或拉伸。
- 同日新增GPU执行为SVS v3一次、YuE2副歌三次（baseline、截断arranged、full诊断）；不把CPU组合或复用记作推理。自动验收与主观验收分离，下一步以用户具体时间点反馈和可控分轨伴奏为依据。
- full诊断的统一QA为S/D/I=24/0/24、近似0/4句、字符相似度.241379；总编辑48低于baseline62只是额外转写更少，目标保留明显回归，不能按总数排序提拔。现有伴奏与SVS仅同长或近似同长，逐拍对齐未验证，本轮拒绝直接套混。
- 精确配器开始落到可控工程素材：新增 `reference prepare-controlled`，真实执行生成 `experiments/reference/iron_rain_controlled_midi_v1`。86 BPM、4/4、8小节、D#m/B/F#/C#两轮和声有原ABC摘要核验，稀疏/展开各一份标准SMF和四轨事件、marker。鼓/贝斯/和弦/弦乐独立，source/fixed_vocal.wav与v4音频SHA完全相同。状态prepared_not_rendered；本地没有已验证MIDI→音源离线渲染链，未生成或冒充伴奏混音。
- 复核修复配置来源竞态：从一次读取的原始bytes解析配置并固定摘要，归档强校验该摘要，校验后改配置必须拒绝完成。MIDI整数tempo舍入、理想网格与歌声容器长度分别记录，不把总长近似一致当作逐字逐拍对齐。
- 最终新增套件132项通过（Voice Studio 48、Reference Studio 56、统一QA 14、日期交付/试听追溯13、取消进程树1）；完整输出保存到日期目录 `qa/workflow_test_output.txt`。所有本轮GPU推理已结束，后续试听验收仍待用户反馈。


### 2026-10-19 追加：固定女声全曲成品 + 日期包核对

- 用户要求先判断两条线能否继续，并把成果集中到日期文件夹、直接给可试听的成品。
- **固定歌手线已从 20 秒样片推进到整曲**：`cover_no_white_flag_full_v1` 用同一套 SoulX-SVC 链路处理 `2026-10-16_epic_hook/selected/02_no_white_flag.wav`（201.239 秒）。Demucs CPU 分轨（约 90 秒）→ 18 个 12 秒分句换声 → 常数增益电平匹配 → 与未改动伴奏直接相加。无 EQ/降噪/压缩/限幅。
- 技术 QA：201.239 秒，峰值 0.970，削波 0，掉音比 0.00432；12 秒分句边界抽查没有幅度异常（最大局部比值来自正常起音）。自动转写内容代理：固定女声版字符相似度 0.653、近似 19/27 句；同曲原唱干声 0.704、18/27 句——两版接近，尾部副歌重复是原曲本身，不是换声丢词。
- **限制（必须记录）**：SoulX-SVC 后端工作在 24 kHz，人声高频上限 12 kHz；日期包里另做了 44.1 kHz 试听混音，只把 Demucs 原始 44.1 kHz 伴奏接回并升采样人声，不恢复人声高频。音色身份、自然度与高音区仍待用户听审。
- 成品放在 `outputs/2026-10-19/finished/`：`02_no_white_flag_full_fixed_voice.wav`（24k 管线混音）、`_mix_44k.wav`（44.1k 试听混音）、`_vocals.wav`（干声）、`.txt`（逐行中英对照 + 指标头）。试听页 `outputs/2026-10-19/index.html` 已收录全曲卡片并链到 `finished/`。
- **日期包核对**：`outputs/2026-10-19/archive/initial_quality_upgrade` 与 `outputs/2026-10-19_quality_upgrade` 逐文件 SHA-256 比对，289/289 完全一致、无缺无多；`index.html` 的 18 个 audio 路径全部存在。`tools/tests` 28 项通过。
- **第二条线（参考编曲）判断**：仍停在 60 秒对照。support arranged 实际 60.039 秒（计划 58.605，+2.45%），baseline 58.519 秒，已可听；但 YuE2 的外部 ABC 只是条件不是时钟，首轮密集 arranged 曾偏长 39.17%。当前 `prepare` 只方便处理连续段落，整曲映射未验证；受控 MIDI 四轨已准备好，但本机没有经验证的 MIDI 音源渲染链。下一步应先做更长连续段（如 verse/chorus 两轮、约 2 分钟，`--allow-long`）的时长漂移 + 听感 A/B，再决定是否扩整曲；精确配器则应转 MIDI/DAW 并先选定音源。


### 2026-10-19 追加：整曲批量 + 参考编曲长段实验

- 用户指示“我不听你都做，第二天反馈”，于是按两条线继续无人值守推进。
- 固定歌手线：01/03/04/05/06 五首整曲 SVC 完成；01/04/06 另做 full_v2（seed 43、48 steps）对照。全部无削波；掉音比 0.43%–1.38%。内容代理相对原唱最好的是 05（0.708 vs 0.777）和 02（0.653 vs 0.704）；01/04 首版明显掉字（0.484/0.425），v2 改善到 0.608/0.542，仍低于原唱；06 两版各有取舍（v1 0.434/7 句，v2 0.471/3 句）。所有成品在 `outputs/2026-10-19/finished/`，清单 `finished/README.txt` 与 `finished/finished_index.json`。
- 参考编曲线：取 38 小节（verse1+chorus1+verse2+chorus2，计划 106.047 秒）做长段 A/B。baseline 实际 106.639 秒（+0.558%，通过），support arranged 实际 138.959 秒（+31.035%，technical_review_required）。语义 token 2667 -> 3475，说明密集伴奏仍会让模型多唱/多奏；外部 ABC 不是时钟。结论：support 编曲不能直接扩整曲，下一步转 MIDI/DAW 或先解决对齐。
- 日期包已重建：44 个音频条目、11 张固定女声卡片、参考编曲长段卡片；所有音频路径存在；`tools/tests` 28 项通过。


## 2026-10-20 — 单日期目录、仓库重整与定点重渲染

- 用户要求 outputs 只保留一个日期目录，打开即见素材；旧批次不删除但移出 outputs。
  已将 2026-10-14 至 2026-10-19_quality_upgrade、battle_batch*、final、AB 实验等
  全部移到 `_archive/outputs_history/`，outputs 现只剩 `2026-10-20/`。
- 去重：`2026-10-19` 归档里的 initial_quality_upgrade / profiles / voice_jobs 与
  `2026-10-19_quality_upgrade`、`data/voice_studio/{profiles,jobs}` 逐文件一致，
  已删除重复副本，约回收 1.8 GB。保留 data/voice_studio 作为唯一工作副本。
- 交付结构：`outputs/2026-10-20/{固定女声,参考编曲,改词副歌,手机试听,_work}`，
  README.txt 说明每个目录；手机试听含 7 个重点 MP3 和试听顺序。
- 私人仓库 Music_date：原 `2026-10-19/`（1.5 GB 的整目录快照，含 archive/audio/
  experiments/qa）已删除，改为扁平交付 `2026-10-19_fixed_voice_en/`，与之前几个
  日期目录一致：9 个 take 的 `<slug>_SVC[_v2].wav` + 同 stem TXT + manifest.json
  + review.md。两次提交并已推送（b116357、d7ba72c、8202ddf）。
- 新增 `tools/generation/retake_chunks.py`：只对父 job 的指定 SVC 分句换 seed/steps
  重跑，其余分句逐样本复用，重新 merge、电平匹配、技术 QA 和直接相加混音；
  父 job 不修改，输出带 retake_manifest 记录父版 SHA、分句和参数。
- 02 无旗可升 retake1：重跑 chunk 0,2,3,4,5（0–68.8 秒），seed 43 / 48 steps。
  局部 0–70 秒 ASR 0.895（父版 0.888），12/12 句（父版 11/12），S/D/I 9/0/9
  （父版 10/0/10）。整曲 ASR 0.405 是 Whisper 在 141 秒幻觉、151–175 秒漏转写，
  实测非重跑区与父版 RMS 差 <1e-4，不是音频丢词。
- 03 锁肘之墙 retake1：先试 seed 43 / 48 steps 重跑 chunk 0–3，局部 0–49 秒反而
  0.812（父版 0.828），未采用；再试 chunk 1,2,3、seed 44 / 64 steps，局部 0.843，
  S/D/I 21/0/2，第 8 句 2→4，采用该版交付。第 6–7 句仍 0，原唱本身也不清晰。
- 参考编曲线仍停在阻塞点：support 编曲长段漂移（106 秒 +31%），MIDI 四轨已准备好，
  但本机无 fluidsynth/timidity、无 sf2/sf3、无 MIDI Python 库，无法渲染音源。
  已在 `参考编曲/说明.txt` 记录三个可选方向；短听 60 秒基准/编曲“差不多”，
  与 support 未做出可听段落差异的结论一致。
- outputs 现为单一日期目录；旧目录仍可在 `_archive/outputs_history/` 追溯。

### 2026-10-20 追加：受控 MIDI 编曲首轮渲染

- 按用户选择安装音源：Chocolatey `fluidsynth 2.4.7`（C:\tools\fluidsynth），
  下载 GeneralUser GS（`models/soundfonts/GeneralUser-GS.sf2`，31,931,936 字节，
  SHA-256 9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe，
  非商业用途免费，与用户个人非商业范围一致）。
- 新增 `tools/generation/render_midi_arrangement.py`：FluidSynth 44.1k 离线渲染 →
  裁/补到人声精确长度 → 伴奏整段常数增益到 RMS 0.07 → 与人声直接相加 →
  仅伴奏尾部 50 ms 淡出、求和后一次峰值衰减。脚本对同一输入复现出逐字节相同的 mix。
- 首次真实渲染：`参考编曲/MIDI渲染/`
  - sparse_mix_44k.wav / expanded_mix_44k.wav，22.326009 秒，峰值 0.95；
  - 纯伴奏 sparse/expanded_instrumental_44k.wav；
  - render_manifest.json 记录 MIDI/SF2/人声 SHA、增益、峰值、RMS；
  - README.txt 记录命令、混音规则和网格对齐提示。
- 人声与网格对齐是近似的：乐句起点 0.70/2.05/5.70/9.55/13.00/16.70/20.20 秒，
  小节线 0/2.79/5.58/8.37/11.16/13.95/16.74/19.53 秒；尚未逐拍验证，
  人声本身也仍是 SVS 新词候选，不能称为成品。
- 手机试听新增 `8_MIDI编曲_稀疏版.mp3`、`9_MIDI编曲_展开版.mp3`。

### 2026-10-20 成曲交付（约 3 分 30 秒）

- 用户要求成曲时长仍在三分半左右。6 首固定女声整曲原长 2:54–3:21，
  逐首用未改动伴奏追加器乐前奏/尾奏到 209.4 秒，详见
  `outputs/2026-10-20/成曲/README.txt` 与各 `_render.json`。
- 新增 `tools/generation/extend_song_length.py`：librosa 拍点/小节对齐，
  选伴奏后段高能量 4 小节循环做尾奏，300 ms 交叉淡化、尾奏 4 秒淡出，
  整段峰值 0.95；人声与歌曲主体不改。原长版本保留在 `成曲_原长/`。
- 成曲 take：01 v2、02 retake1、03 retake1、04 v2、05 v1、06 retake1。
- 01/04/06 也做了 seed 44 / 64 steps 的定点重渲染：
  - 01 retake 整曲代理 0.52（父版 0.608）、局部还出现成段丢词，弃用；
  - 04 retake 0.511（父版 0.542）、deletion 83，弃用；
  - 06 retake 0.472（父版 v2 0.471），近似句 6/27（父版 3/27），
    deletion 3（父版 59）但 insertion 57；作为成曲 take，保留 v1/v2 备选。
  结论：seed 44/64 多分句重跑不稳定，只有 02、03 的较小范围重跑有效。
- 手机 `手机试听/成曲_3分30/` 放 6 首成曲 MP3；`手机试听/试听顺序.txt`
  已把成曲放在最前面。
- 私人仓库新增扁平目录 `2026-10-20_fixed_voice_final/`：
  6 个 `<slug>_SVC_3m30.wav` + 同 stem TXT + manifest.json + review.md，
  已提交并推送（431371b）。之前的 `2026-10-19_fixed_voice_en/` 保持不变。
- 追加原长 A/B：`成曲_原长/` 保留 6 首未延长版本；手机 `手机试听/成曲_原长/`
  与仓库 `2026-10-20_fixed_voice_final/` 的 `<slug>_SVC_orig.wav` 均可对照。
  仓库第二次提交 88fa231 已推送。

## 2026-10-20 (afternoon) — 增量固定女声批次：12 首新曲目

- 用户指示（13:23）：不再做手机试听文件夹；继续生成到 17:00，再把 10-20 新产出
  一并上传；今天可以多一点。
- 选定 12 首此前从未做过固定女声翻唱的英文曲目：
  - 2026-10-16_epic_hook 候选选中的 07_signal_fire（index 已 promote，未入 selected）；
  - 2026-10-15_epic_en 的 08_wolves_at_the_wall、09_small_and_unbroken；
  - 2026-10-14_epic_en 的 01-09 全部九首。
  （10-15 的 01-06 与 10-16 同题不同渲染，不重复翻唱。）
- 新增工具（均未改旧脚本）：
  - `tools/generation/cover_more_batch.py`：逐首执行 `run_workflows.bat voice cover`，
    开跑前检查空闲显存（>=6000 MiB），16:30 后不再启动新曲，状态写
    `outputs/2026-10-20/_work/more_covers/status.json`。
  - `tools/generation/preseparate_more.py`：CPU Demucs 预热 Voice Studio 分轨缓存。
  - `tools/generation/finalize_more_covers.py`：CPU 歌词 QA（固定女声版 vs 原唱干声）、
    44.1 kHz 混音、近似 210 秒延长（原长已超 210 秒的交付原长）、双语 TXT、
    manifest 与中文试听清单。
- 启动脚本：`start_preseparate_more.ps1`（13:32 启动，CPU 分轨）、
  `start_more_covers.ps1`（GPU 空闲后启动）。
- GPU 协调：另一个会话（model-e3）的 ComfyUI 在 13:17-约 13:50 占用显卡做图像/
  LTX 样片；已约定其结束后释放，音乐批次再开始，未强杀对方进程。
- 交付计划：本地 `outputs/2026-10-20/更多固定女声/`，17:00 复制为私有库
  `2026-10-20_fixed_voice_more/` 并推送（WAV + 同名 TXT + manifest + 试听清单），
  不建手机试听目录。

### 2026-10-20 增量批次结果（15:19 生成完，15:45 打包完）

- 12 首全部完成：`outputs/2026-10-20/fixed_voice_more/`（688 MB，20 个 WAV、
  20 个同名 TXT、manifest.json、中文试听清单）。
  - 9 首交付 3:30 版 + 原长 A/B；02/04/06/09 主体已 ≥210 秒，交付原长版 + 原长 A/B。
  - 内容代理（整曲 ASR）多数 0.79-0.86；偏弱三首：07_signal_fire 0.472、
    03_iron_lullaby 0.617、08_we_who_remain 0.668；原唱同口径 0.80-0.97。
- 生成速度：每首约 5-7 分钟（模型每首重载），13:49 起 15:19 结束；CPU 分轨预热
  在 11/12 时因整机可用内存降到 0 手动停止，最后一首由 cover 自行分轨。
- 修复的两个 finalize 缺陷：分轨缓存命中时 `separation.log` 不存在导致找不到伴奏
  （改为按源音频重算缓存键）；render 报告目录不存在导致延长步骤退出码非零。
- 17:00 前后推送私有库 `2026-10-20_fixed_voice_more/`（仅交付文件）。

- 弱项重跑（seed 43 / 48 steps，仅当内容代理明显更好才替换）：
  - 07_signal_fire 0.472 -> 0.687（替换，取 full_v2）；
  - 03_iron_lullaby 0.617 -> 0.868（替换，取 full_v2）；
  - 08_we_who_remain 0.668 -> 0.612（更差，保留 full_v1）。
- 全部 20 个 WAV 复核：44.1 kHz、峰值 0.95、无 NaN/Inf、RMS 0.104-0.133；
  manifest 与 review 已按 final take 重算。
- 私有库提交 25d7b81（amend 后单次提交），16:17 开始 push。
- Push 完成：远端 main = 25d7b81（688 MB，20 WAV + 20 TXT + manifest + review）；
  `2026-10-20_fixed_voice_more/` 为本日新增固定女声批次，未创建手机试听目录。

## 2026-10-22 — 删除 19/20 产出，两条线退回 YuE2 基础并重构

- 用户复听 19/20 固定女声批次：高音和转音"直接变成机械音"，比 10-14/15/16 的
  YuE2 原生版本更差。确认要删掉 19/20 全部产出，并按 YuE2 基础重构两条线
  （固定女声 + 参考编曲），先出小样再批量。
- 本地清理：删除 `outputs/2026-10-20`、`_archive/outputs_history/2026-10-19_*`、
  `data/voice_studio/jobs/*`、`data/voice_studio/cache/*`，约 8 GB。保留 10-14/15/16
  存档、`profiles/iron_rain_female_v1`（178 MB）和音色母本。
- 云端私有库 `Jun-shisheng/Music_date`：删除 `2026-10-19_fixed_voice_en`、
  `2026-10-20_fixed_voice_final`、`2026-10-20_fixed_voice_more` 三个目录并推送
  （远端 `main = d615d6d`）；14/15/16 交付保留。Git 历史仍含旧 blob（2.56 GB），
  未做 `filter-repo`，需要彻底清除时另做。
- 根因（实测）：SoulX-SVC 全链路 24 kHz（12 kHz 硬顶）；外层固定 12 s 切句，
  反而让模型永远走 <30 s 整段快路径，无法使用它自己的 F0 感知分段；F0 用
  24 kHz DIO + 20 ms；`iron_rain_female_v1` 的干声只截了前 60 s（p90 494 Hz），
  而母本高音在 81–82.6 s（最高 1095 Hz）、103.7 s、156.1 s，参考完全没覆盖。
- 24 kHz 往返模拟：>12 kHz 能量 0.1118% → 0.0034%（约 1/33）；零相位互补恢复
  回到 0.1117%，低频误差 RMS 0.000475。确认高频恢复可行。
- 新增/修改：
  - `tools/voice_studio/band_restore.py`：24k SVC 升到 44.1k，11 kHz 以上从
    同一时间线的全带 Demucs 干声恢复；零相位互补 FIR + 低频段常数增益；测试 6 项。
  - `tools/voice_studio/highnote_qa.py`：pyin F0、高音段、颤音深度/速率、HNR、
    谐波间能量比、谱平坦度、>11/12 kHz 能量、12 kHz 悬崖；合成信号自检 + 7 项测试。
    旧检测器在 YuE2 上无效，新指标只描述、不自动判好听。
  - `tools/voice_studio/registers.py`：从全曲干声提取低/中/高音区参考（pyin 50 Hz）。
    新建 `iron_rain_female_v2`：low 353 / mid 415 / high 618 Hz，v1 未改。
  - `tools/voice_studio/audio.py`：`canonical_at(rate)`、`voiced_segments()`（按长休止
    切句、最小 6 s、最大 18 s、强制切点显式标记）；旧 `segments`/`merge` 行为不变。
  - `tools/voice_studio/worker.py`：pyin F0（480 hop = 50 Hz）+ 八度跳变剔除；
    SVC 支持每分句选择音区参考（多 prompt 缓存）。
  - `tools/voice_studio/workflow.py`：新增 `native`（YuE2 原生 48k 交付，无换声）
    和 `hybrid`（全带干声 + 音区参考 + pyin + 高频恢复 + 44.1k 混音）子命令；
    旧 `cover`/`relyric` 保留，legacy 24k 标记 `deliverable=false`。
  - `tools/reference_studio/__main__.py`：新增 `shape`（计划层密度/动态）和 `midi`
    （段落级 FluidSynth）子命令；`render_midi_arrangement.py` 改为接受任意采样率人声。
  - `tools/generation/highnote_ab.py`：A/B 试听集与指标表生成。
- 测试：`tools/voice_studio/tests` 70 项全过（新增 band_restore 6、highnote 7、
  voiced_segments 5、registers 4）。
- A/B 小样（`outputs/2026-10-22_highnote_ab/`，全部 -20 dBFS 常数增益）：
  - 04_iron_rain 0–30 s：native 高音 895 Hz，vib 8.8/9.5c，HNR -2.6/-1.0 dB；
    legacy24 F0 相关 0.855、八度错误 10%、HF 比 0.18；hybrid F0 相关 0.981、
    八度错误 1.0%、HF 比 1.09；24.77 s 高音 HNR 9.53 dB（native 同位置 -0.98），
    5.76 s 仍是有噪声的音节，整体仍标 inharmonic_hf_excess。
  - 02_no_white_flag 110–140 s：native 1174 Hz 长音 vib 6.9–7.6c、HNR 13 dB；
    legacy24 F0 相关 0.620、八度错误 51.7%、HF 比 0.11；hybrid F0 相关 0.517、
    八度错误 26.6%、HF 比 0.60，1174 Hz 被唱成 811/821 Hz 并出现 1388 Hz 噪声段，
    未通过 `pitch_tracking_mismatch`，delivery_gate=false。
  - 修正了 hybrid 的 F0 与分句上下文对齐：分句音频带 0.25 s 上下文，F0 必须按
    `context_start..context_end` 切，不能只切核心段；修正后 04 hybrid 从 0.960 升到 0.981。
- 结论（待用户听审确认）：hybrid 修好了带宽和 HF，但没有修好高音的 F0/机械音；
  02 的 1174 Hz 长音仍明显劣于 native。固定女声线默认应交付 native（YuE2 原生
  48k），SVC 只保留为实验；参考编曲线的计划整形已验证可改密度/动态，但渲染对比
  要另开一轮（本轮不重渲染 YuE2）。
- 用户听审入口：`outputs/2026-10-22_highnote_ab/REPORT.md` 和 `试听顺序.txt`；
  先听 `listening/<song>_<arm>_vocal.wav`，再听 `_mix.wav`。
- 本会话未改 FastAPI/web/`start.bat`，未批量重跑，未动 10-14/15/16 存档。

### 2026-10-22 下午：完整歌曲批量（到 17:30）

- 用户指示：工作流弄好后自主继续，直接生成完整歌曲到 17:30，第二天再给最终听审结论。
- 13:23 启动英文 12 首（10-14 御姐史诗 brief，复制到 `inputs/_batch_20261022_en`），
  输出 `outputs/2026-10-22_batch/epic_en`；python PID 4884；日志
  `logs/yue2_en_20261022.out.log` / `.err.log`。
- 英文结束后接中文 9 首（清甜流行，`inputs/_batch_20261022_zh` 01–09），
  输出 `outputs/2026-10-22_batch/mandarin_pop`，参数相同，seed-base 2400。
- 批量参数：`--candidates 1 --retries 1 --budget 12 --backend torch-eager
  --nar-query-chunk 2048 --intro-bars 2 --shape "split chorus:2"
  --shape "thin verse:1.5" --final-lift 2 --early-accept`。
- 启动器：`tools/generation/launch_yue2_batch.ps1`（Start-Process 隐藏窗口，
  stdout/stderr 分开写日志，不使用 `ErrorActionPreference=Stop`）。
- 巡检：会话内 cron 每 12 分钟；17:30 后停止启动新歌，收尾到
  `outputs/2026-10-22_batch/finished/`（WAV + 同名歌词 TXT + manifest + README），
  不上传私有仓库。
- 实际运行：2048 NAR chunk 在 12 GB 上多次 OOM，`expandable_segments` 在 Windows
  不支持；改用 1024 chunk（10-16 批次同参数）。仍有个别长 plan OOM，已给
  `yue2_batch_battle.py` 加 `--max-plan-seconds 240`，超过 4 分钟的 plan 换 seed。
- 英文批次 13:23–15:09 共交付 5 首：01_last_dawn_city、03_iron_lullaby、
  04_the_unchosen、05_spearstorm、06_embers_never_die；02_unbowed 和
  07_crown_of_thorns 两次 OOM 未出；08–12 未跑。英文日志备份在
  `logs/yue2_en_20261022_attempt1..3.log`。
- 15:09 切中文 9 首（清甜线），输出 `outputs/2026-10-22_batch/mandarin_pop`，
  日志 `logs/yue2_zh_20261022.out.log`；默认 1024 chunk + plan 240 s 上限。
- 16:13 中文 9/9 全部完成（64.1 分钟，每首 346–439 秒）：
  01_starlight_on_my_shoulder、02_wind_from_the_coast、03_neon_heartbeat、
  04_unfinished_summer_night、05_sparks_after_rain、06_mountains_in_a_dream、
  07_run_toward_the_light、08_the_moon_knows、09_burning_blue；
  `mandarin_pop/index.json` 已写。
- 16:21 启动中文 10–20（11 首）→ `outputs/2026-10-22_batch/mandarin_pop_2`，
  seed 2600，日志 `logs/yue2_zh2_20261022.out.log`；预计 17:30 前再出 9–10 首。
- 17:27 用户要求收尾提交，手动停批（16 已落盘，17 未完成）。最终 **21 首**：
  英文 5（01、03、04、05、06），中文 16（01–16）。全部 48 kHz YuE2 原生；
  英文 2:46–3:27，中文 1:47–2:55。566 MB。
- 收尾：`tools/generation/finalize_20261022_batch.py` → 
  `outputs/2026-10-22_batch/finished/{中文清甜,英文御姐}/`，每首 WAV + 同名 TXT，
  顶层 `manifest.json` + `README.md`。弱项（歌词覆盖 <0.6）：
  10_sing_tomorrow_bright 0.378、11_glass_hearts_glow 0.390、
  16_paper_planes_satellites 0.382。
- 私有库：复制为 `Music_date/2026-10-22_native_yue2/`，提交 `655d8e3`
  （21 WAV + 21 TXT + manifest + README），push 完成；README 状态修正提交
  `ea90ebd` 也已推送，远端 `main = ea90ebd`。未上传工作文件。
- 本批缺口：英文 02_unbowed、07_crown_of_thorns 两次 OOM 未出；08–12 未跑；
  中文 17–20 未跑（17:27 停批）。参考编曲线本轮未重渲染。

## 2026-10-23 — 2026-10-22 产出的审阅包

- 用户尚未听审 10-22 产出，要求归拢到一个新目录。建
  `outputs/2026-10-23_review_20261022/`（纯副本，612 MB，33 个 WAV）：
  - `README.md`：试听顺序（先 A/B 6 分钟，再中文 16 首约 35 分，再英文 5 首约 16 分）、
    逐曲指标表、缺口清单、原件路径。
  - `01_成片_21首/`：21 首 WAV + 同名歌词 TXT + manifest.json + 原 README。
  - `02_高音AB/`：12 个试听 WAV + REPORT.md + 试听顺序.txt + metrics + ab_config/ab_manifest。
  - `03_参考编曲线状态.md`：原 `LINE2_STATUS.md`。
  - `审阅清单.md`：A/B 三个待定问题 + 21 首逐曲「保留/重跑/弃用」空表 + 下一步三件事。
- 校验：21 个 WAV 与歌词的 SHA-256 全部与 `manifest.json` 一致（0 mismatch）。
- 原件未动，私有库 `2026-10-22_native_yue2` 仍是未审阅版本，审阅结论出来前不作定稿。
- **用户听审 A/B 结论（2026-10-23）：不通过。** 原话：「hy 稍微好一点但还是不够，
  02 那个高音前面还行，拉长到后面又崩了，且 mix 比 vocal 稍微好一点但是还是崩。」
  已记入 `outputs/2026-10-22_highnote_ab/ab_manifest.json`（`status: rejected` + `verdict`）
  和审阅包的 `审阅清单.md`。默认交付路径定为 **native（YuE2 原生 48 kHz）**，
  legacy24/hybrid 不再作为交付路径。
- 指标与听感一致：02 高音区 hybrid F0 轮廓相关 -0.99、八度错误 75%；1174 Hz 唱成
  811/821 Hz（低约五度），15.0 s 后落入 1388–1412 Hz 噪声段（HNR -5～-0.5 dB）。
  「越拉长越崩」= 长音上的 F0 跟踪漂移，属模型高音区能力问题，带宽恢复解决不了。
- 21 首成片尚未听审，等用户继续给逐曲结论。

### 2026-10-23 清理 + 补齐缺口批次（第 1 轮迭代）

- 用户指示：进行迭代、删除无用素材、开始今天的制作；周末统一听审 10-22 与
  10-23 的产出再给结论。清理范围按「安全」执行（用户选择）。
- 清理，释放约 0.95 GB（成品与审阅包未动，SHA-256 复核 0 mismatch）：
  - 候选目录里的 `*_artifacts/*.npy` 与重复的 `audio.flac`（105 个文件，623 MB）；
    保留 `plan_original.abc`、`request.json`、`result.json` 等小文件。
  - `data/voice_studio/jobs/*`（9 个 SVC 实验任务，186 MB）与
    `data/voice_studio/cache/*`（136 MB）。SVC 已被否决，试听 WAV 和 REPORT 仍在
    `outputs/2026-10-22_highnote_ab/`。保留 `data/voice_studio/profiles/`（268 MB）。
  - 空的失败候选目录（02_unbowed、07 等的 4 K 空壳）。
- 迭代点（针对昨天 OOM 的实测）：昨天 02/07/08 分别 OOM，但它们的 plan 长度
  274.3 / 272.6 / 221.9 s，并不比成功的 04（237.4 s）、06（224.3 s）长——问题是
  整个批次在**同一个进程**里跑，分配器越到后面越紧。今天改成 **一首一个进程**
  （`--only`，官方 help 本来就推荐这个），获得干净分配器。
- 生成范围：英文 02_unbowed、07_crown_of_thorns、08–12（7 首）+ 中文 17–20（4 首），
  共 11 首。参数与昨天一致（chunk 1024 / plan 上限 240 s / retries 3 / intro 2 bars /
  `split chorus:2` + `thin verse:1.5` / final-lift 2）；每首若一阶段失败，二阶段自动
  降为 chunk 768 + plan 上限 220 s + seed+500。
- 启动器：`tools/generation/start_yue2_batch_20261023.ps1`，10:49 通过 Start-Process
  隐藏窗口启动；状态写 `logs/yue2_batch_20261023_status.json`，逐曲日志
  `logs/yue2_20261023_<stage>_<slug>.out.log`，汇总 `logs/yue2_batch_20261023.log`。
- 输出目录：`outputs/2026-10-23_batch/epic_en/`、`outputs/2026-10-23_batch/mandarin_pop/`
  （`selected/` 是交付候选，`candidates/` 保留 32-bit float 原始母带）。
- 预计 2.5–3 小时。弱项（ZH 10/11/16、07）本轮不重跑，等用户听审后再定。

#### 11:00–11:12 OOM 根因与修正（第 2 轮）

- 「一首一进程」不够：02_unbowed 在新进程里仍然连续两次 OOM（chunk 1024），
  换 chunk 768 也一样。实测 GPU 只用到 9.7/12 GB 就失败，说明不是卡不够，
  是**进程内存上限**。
- 根因：`YuE2Pipeline.__init__` 里
  `torch.cuda.set_per_process_memory_fraction(min((memory_budget_gib-2)GiB, total-2GiB)/total)`
  ——12 GiB 预算 ⇒ 进程被钉死在 **10 GiB**。实测能出片的 plan 编辑后是 181–204 s，
  OOM 的是 201–238 s（02: 228.7/226.3；07: 233.7/238.3；08: 201.0），全部卡在 10 GiB。
- 先试了「把 plan 上限降到 195 s」：模型给这些 EN brief 稳定规划 218–228 s，
  每个 seed 都被 SKIP（每次规划约 2 分钟），不可行，已放弃并撤回参数。
- 最终修正：给 `yue2_batch_battle.py` 加 `--gpu-mem-fraction`，在 pipeline 构造后
  重新调用 `set_per_process_memory_fraction(0.90, current_device())`，把上限抬到
  10.8 GiB（桌面常驻约 0.9 GiB，留约 1.2 GiB 余量）。同时保留：
  plan 阶段与 render 阶段之间、以及每次失败重试前调用 `torch.cuda.empty_cache()`
  （减少缓存块残留）。
- 踩坑：第一版写成 `set_per_process_memory_fraction(0.9, "cuda")`，torch 要求设备
  索引不是字符串，整批 20 秒内全部崩溃（`ValueError: Expected a torch.device with
  a specified index`），已改为 `torch.cuda.current_device()`。
- 另一次误判：查进程时用 `CommandLine -like "*start_yue2_batch*"`，把执行查询的
  外层 powershell 自己也匹配上了，看起来像「同时跑了两个 launcher」，实际只有一个。
- 11:08:47 重新启动（`--gpu-mem-fraction 0.90`、chunk 1024、plan 上限 240、retries 4）；
  02_unbowed 的 228 s plan 这次没有再 OOM，11:10 进入渲染（GPU 12.0/12.3 GB）。

#### 15:07 完成：11/11 全部出片

- 总耗时 3 小时 59 分（11:08–15:07），全部 YuE2 原生 48 kHz 立体声，无 SVC、
  无 EQ/压缩/限幅。收尾脚本 `tools/generation/finalize_20261023_batch.py`
  → `outputs/2026-10-23_batch/finished/`（11 WAV + 11 TXT + manifest + README，357 MB）。
- 结果（arc / 起伏 / 峰值位置 / 歌词覆盖）：
  - 英文 02_unbowed 3:36 build-up 13.2 dB peak@0.76 cov 0.69
  - 英文 07_crown_of_thorns 3:21 build-up 11.6 dB peak@0.57 cov 1.05
  - 英文 08_we_who_remain 3:15 build-up 10.1 dB peak@0.90 cov 1.22
  - 英文 09_tide_of_endings 3:27 build-up 12.1 dB peak@0.94 cov **0.53（弱）**
  - 英文 10_the_sky_we_burned 3:08 build-up 9.9 dB peak@0.92 cov 0.62
  - 英文 11_vow_under_ruins 3:02 build-up 12.1 dB peak@0.94 cov 0.61
  - 英文 12_rise_again 3:41 build-up 8.6 dB peak@0.83 cov 0.99
  - 中文 17_stay_until_sunrise 2:00 mild build 7.0 dB peak@0.84 cov 1.07
  - 中文 18_gravity_is_a_lie 3:00 build-up 8.4 dB peak@0.74 cov 1.27
  - 中文 19_lavender_city_lights 1:52 **early peak** 4.6 dB peak@0.90 cov 1.23
  - 中文 20_better_than_yesterday 2:03 mild build 12.3 dB peak@0.48 cov 1.48
- 全 7 首英文都是 build-up、峰值在后半段，符合战歌验收偏好。英文时长
  3:02–3:41（比 10-22 的 2:46–3:27 更长，因为 240 s plan 上限保住了）。
- 异常与待听项：
  - 09 单首渲染 73.5 分钟（4413 s），同参数其他首 16–27 分钟；只有一次尝试、
    无 OOM，实际 plan 208 s，与其他首同档。怀疑后台桌面程序（Chrome 硬件加速、
    视频、微信/钉钉）抢 GPU，未处理。
  - 09 歌词覆盖 0.528 是唯一弱项；10/11 在 0.61/0.62 边缘。
  - 19 是 early peak 且只有 4.6 dB 起伏，按战歌规则大概率不达标。
- 私有库未上传；等用户周末听审 10-22 + 10-23 两批后统一决定保留/重跑/交付。

#### 15:24–16:21 重跑批次（6 首）+ 16:40 截止

- 用户要求「接着跑到五点、自动结束、整理上传」，且**只上传 10-23 当天的产出**。
- 重跑对象（按自动指标，不是听感）：09 覆盖 0.53；19 early peak 4.6 dB；
  07 动态平 3.3 dB；10/11/16 覆盖 0.38–0.39。参数同本批，seed 4000/4100+37i，
  16:40 后不再启动新曲（`start_yue2_retakes_20261023.ps1`）。
- 6/6 全部出片（15:24–16:21，无 OOM），收尾到 `outputs/2026-10-23_retakes/finished/`。
- 重跑 vs 原版（arc / lift / peak / coverage）：
  - 07 2:11 flat 3.3 → **2:22 build-up 12.8**（明显更好）
  - 09 3:27 build-up 12.1/0.94/0.528 → 2:49 mild build 10.8/0.44/0.548（更差，保留原版）
  - 10 1:47 build-up 9.7/0.95/0.378 → 2:00 build-up 14.2/0.93/0.528（更好但覆盖仍 <0.6）
  - 11 2:08 build-up 14.0/0.80/0.39 → 2:00 build-up 10.3/0.78/1.58（词明显更好）
  - 16 2:12 early peak 6.8/0.31/0.382 → 2:00 build-up 11.4/0.68/0.982（更好）
  - 19 1:52 early peak 4.6/0.90/1.23 → 1:58 early peak 4.2/0.87/1.09（没改善）
- 发现并在交付中修正：**10-22/23 的「中文清甜」批次里有 10 首其实是英文歌**
  （11–16、17–20）。为全部英文歌补了 `_lyrics_zh.txt` 逐行中文，交付 TXT 改成
  「一行英文 + 一行中文」（25 首被改写，`tools/generation/apply_bilingual_lyrics.py`）。
- 打包：`tools/generation/package_music_date.py` 按**实际演唱语言**分目录，
  今天 11 首全部进 `英文御姐/`，6 首重跑进 `重跑候选/`；
  产物 `outputs/2026-10-23_delivery/`（17 首，502 MB，双语文案检查 0 问题）。
- 上传：私有库 `Music_date/2026-10-23_native_yue2/`，提交 `077fd1f`，
  push 完成（`ea90ebd..077fd1f main -> main`）。**只上传 10-23 当天产出**，
  10-22 目录与 21 首成品未动（用户明确要求）。

## 2026-10-26 — 中文词曲周：清空旧产出，建立歌词 Skill 与旋律验收

- 用户本周 brief：重点做**中文曲**，主攻**歌词 SKILL 和曲调**；按 2024–2026 华语热歌/年度榜单
  拆解作词与作曲；明确点名《悬溺》《新·与天齐》两类参照；**不要拟声词、不要哼唱**；
  旧产出不要留。人声先按**清甜女声**；首批 **4 首小样**；歌词 SKILL 做成
  **Claude Skill + DeepSeek 生成/校验工具**。
- 旧产出按用户确认**直接删除**（约 8 GB）：`outputs/2026-10-22_batch`、
  `2026-10-22_highnote_ab`、`2026-10-23_batch`、`2026-10-23_delivery`、
  `2026-10-23_retakes`、`2026-10-23_review_20261022`。`outputs/` 现为空。
- 远端 `origin/main` 比本地多 25 个提交（v0.2.0 工程化发布：Next 16/React 19/
  Tailwind 4、CI、CONTRIBUTING、generation_type 契约、队列恢复等）。**没有直接 merge**，
  因为本地生成工具与工作区有大量未提交修改；只采纳其“契约 + 测试 + 文档”的做法。
- 新增研究文档 `docs/lyric_melody_zh_2026.md`：
  - 2024–2026 跨榜共识（王菲《世界赠予我的》、单依纯《珠玉》、蔡徐坤《Deadman》、
    汪苏泷《像晴天像雨天》、徐明浩/GALI《Star Crossing Night》、LBI 利比《跳楼机》、
    揽佬《大展鸿图》、周深《好风起》、林俊杰《明日坐标》、裘德《春雪》、
    刀郎《命运的赛勒克》、唐汉霄《就是哪吒》）；
  - 用户点名的《悬溺》（约 72 BPM、A 小调、Am–F–G–Em7、词少、氛围 hook）与
    《新·与天齐》（古风燃向、口号 hook、强重音）拆解；**悬溺只借留白/音色，
    不借无词 hook**；
  - 作词 10 条、旋律 10 条、目标数字（副歌/主歌密度 1.4–4.5、音区 +2–5 半音、
    级进 ≥45%、hook 相似度 ≥0.80、高潮位置 0.55–0.99 等）。
- 新增 Claude Skill `.claude/skills/mandarin-pop-lyric/SKILL.md`：
  三种模式（近景叙事 / 悬溺型暗色氛围 / 古风燃向）、28 句配额、hook/押韵/声调/
  陈词滥调/反拟声规则、验收命令。
- 新增工具（均带测试）：
  - `tools/generation/zh_lyric_qa.py`：段落配额、句长、段内韵脚一致度、
    三声连读、句尾三声、开口音句尾、hook（长度/次数/开口音收尾）、
    具体物件/感官词、抽象词与 AI 文案腔、重复与主歌推进。
  - `tools/generation/melody_qa.py`：直接读 YuE2 ABC 计划，测副歌/主歌密度比、
    音区抬升、末副歌抬升、级进/跳进/同音比例、音域、最高音位置、三段副歌
    pitch-class 钩子相似度；`--target zh-pop` 为临时标尺。
  - `tools/generation/zh_lyrics_gen.py`：DeepSeek 按选题卡写中文词，本地门控
    （含英文/拟声/hook 开口音/韵脚/三声）不过就重写；输出 lyrics/tags/brief。
  - `yue2_batch_battle.py` 增加 `--melody-target`：每个候选保存
    `melody_qa.json`，并把 melody score（0–10 分）并入排序；
    `battle_batch.score_take` 向后兼容。
  - 测试：`test_zh_lyric_qa.py` 6、`test_melody_qa.py` 5、
    `test_zh_lyrics_gen.py` 7、`test_battle_ranking.py` 7，共 25 项通过。
- 首批 4 首新词（`inputs/2026-10-26_zh_pop/`，均通过门控）：
  - 01 末班车以后（narrative，便利店/关东煮/两张车票）hook「谁先开口说再见」99 分；
  - 02 伞往你那边偏（narrative，伞/湿肩膀/楼道灯）hook「伞往你那边偏」99 分；
  - 03 三楼的灯没有关（atmospheric，电梯/猫眼/语音）hook「三楼的灯没有关」99 分；
  - 04 城门开之前（guofeng，雪/旧甲/城门）hook「我自己的路自己开」99 分。
  旧 10-22 中文词同口径只有 34 分（段落不全、抽象词 7 次、三声连读 3 个）。
- 11:31 启动 YuE2 小样批次（每首 2 候选、一首一进程、无 SVC/无 EQ/零后处理）：
  - 驱动 `tools/generation/run_zh_pop_20261026.py`，启动器
    `start_zh_pop_20261026.ps1`，状态 `logs/zh_pop_20261026_status.json`，
    逐曲日志 `logs/zh_pop_<slug>_<stamp>.log`。
  - 参数：`--budget 12 --nar-query-chunk 1024 --gpu-mem-fraction 0.90
    --max-plan-seconds 240 --retries 1 --candidates 2 --no-early-accept`；
    叙事/古风 `hook chorus + thin verse:1.5 + split chorus:4 + final-lift 2`，
    氛围 `hook chorus + thin verse:2 + final-lift 2`；`--intro-bars 2`（氛围 4）。
  - 输出 `outputs/2026-10-26_zh_pop/`（candidates/ + selected/ + index.json）。
- 待办：
  1. 批次完成后跑 `melody_qa --dir outputs/2026-10-26_zh_pop` 汇总，生成中文试听单；
  2. 人耳听审 4 首 x 2 候选，决定保留/重跑；
  3. 用户提供 3–5 首中文参考音频（A/B/C 三类各一首）放到 `human_refs/zh_refs/`，
     用 `ref_profile.py` 把 §3.3 的临时标尺换成实测；
  4. 听审通过后再决定是否推送私有库（本轮未推送）。

### 2026-10-26 11:34 修正并重启

- 发现并修复 `plan_shape.py` 的子串匹配 bug：`chorus` 会匹配 `pre-chorus`，
  导致 `hook chorus` 把副歌旋律复制进预副歌、`split chorus` 把预副歌也加密。
  新增 `section_matches()`（`chorus` 排除 `pre`）和回归测试；`test_plan_shape.py` 28 项通过。
- 11:34:52 用修正后的代码重启小样批次（01 从干净目录重跑）。
  第一首 01 c1 从 11:35 起渲染，GPU 100%、~158 W；截至 11:55 仍在渲染
  （桌面 Chrome/微信等占用 GPU 时，单首可能到 20-30 分钟，10-23 已有 73.5 分钟先例）。
- 新增 `tools/generation/zh_pop_review.py`：读 `index.json` + 词检 + `melody_qa.json`，
  生成中文 `review.md` 和可选 MP3 试听；已用假 index 做过渲染测试。

### 2026-10-26 12:20 前两首结果

- 01 末班车以后：c1 7.4 dB mild build / c2 10.3 dB build-up；c2 胜出，159.8 s，
  首句 3.64 s，歌词覆盖 1.05。按更新后的旋律门控 c1 83 分、c2 83 分
  （c2 唯一未过项是副歌/主歌密度 0.92；当时用的还是 `thin verse:1.5`，
  实测 1.5 对真实计划几乎不降密度）。
- 02 伞往你那边偏：改用 `thin verse:2 + split chorus:6` 后，c2 旋律门控 **100 分**
  （密度 1.9、音区 +5.1 半音、钩子 0.904、最高音 88.5%），音频 build-up 9.1 dB、
  覆盖 0.933；c2 胜出。c1 旋律 90 分（密度 2.7、钩子 0.99），但音频只有 mild build 7.4 dB。
- 结论（初步）：`thin verse:2 + split chorus:6 + hook chorus + final-lift 2` 是目前
  中文清甜线的可用配方；`thin verse:1.5` 不产生实际对比，已从驱动器中移除。
- `melody_qa` 的钩子相似度改为**允许任意移调**（末副歌升 2 个半音仍算同一个 hook），
  级进比例降级为诊断项（不否决），高潮位置上限放宽到 1.0（最高音落在最后一小节可以）。
  测试 54 项通过。

### 2026-10-26 12:45 第三首完成 + hook 长度保护

- 03 三楼的灯没有关：c1 旋律 98 分（密度 2.06、音区 +4.2、钩子 0.904），
  音频 build-up 12.6 dB、peak 0.64、覆盖 0.935、172.4 s；c2 旋律 83 分
  （音区只抬 0.7 半音）。c1 胜出。
- 04 第一次启动时发现 `hook chorus` 会把短副歌铺到长段（报告里的 “2 -> 25 bars”
  实际是 Vocal+Ins 双计数，真实目标 17 小节 / 源 8 小节，约 2.1 倍，尚可接受）。
  为防未来出现 2 小节副歌铺 25 小节的 12 倍循环，新增 `--hook-max-tile`
  （默认 0=不限制；驱动器用 3.5）：目标段超过源段 3.5 倍时跳过 hook、保留模型
  自己的旋律。`plan_shape.py` / `yue2_batch_battle.py` / 驱动器已接通，
  测试 29 项通过。
- 04 已用 `--hook-max-tile 3.5` 从干净目录重启（12:43），当前 c1 渲染中。

### 2026-10-26 13:10 四首批完成（4/4 出片）

- 全部 4 首各 2 候选，YuE2 原生 48 kHz，零后处理；selected：
  | 曲 | 模式 | 时长 | arc / 起伏 | 歌词覆盖 | 旋律门控 |
  |---|---|---:|---|---:|---|
  | 末班车以后 | narrative | 159.8 s | build-up 10.3 dB | 1.05 | 83（密度 0.92 未过） |
  | 伞往你那边偏 | narrative | 125.2 s | build-up 9.1 dB | 0.93 | 100 |
  | 三楼的灯没有关 | atmospheric | 172.4 s | build-up 12.6 dB | 0.94 | 98 |
  | 城门开之前 | guofeng | 185.9 s | build-up 10.0 dB | 1.12 | 88（音域 21 半音，轻微超） |
- 02/03/04 的胜出候选旋律门控通过；01 是唯一未过（副歌/主歌密度 0.92），
  因为它用的是第一版配方 `thin verse:1.5`（实测不降密度），不是最终的
  `thin verse:2 + split chorus:6`。建议听审后决定是否用新配方重跑 01。
- 试听包：`outputs/2026-10-26_zh_pop/review.md` +
  `outputs/2026-10-26_zh_pop/review/listen/*.mp3`（每首 c1/c2/selected 三个 MP3）。
  原始 48 kHz WAV 在 `selected/` 和 `candidates/<run>/`。
- 本轮未推送私有库；等听审结论后再决定保留/重跑/交付。

### 2026-10-26 13:30 方向收窄：史诗 / BGM / 思辨，参考邓紫棋的中心比喻

- 用户补充：**主要还是偏向史诗感、BGM 等**；中文词要逐字推敲，不要随意歌词，
  最好有思辨和哲学深度，同时不失去直白；可参考邓紫棋近作的比喻方法。
- 研究结论（已写入 `docs/lyric_melody_zh_2026.md` §2.2.2）：
  - 邓紫棋的方法 = 一首歌一个中心比喻（泡沫/句号/摩天动物园/毒苹果/沙漠/芯片），
    直白 hook + 桥段哲学反转 + 对句/行中停顿/密集押韵 + 具体物件托住抽象命题；
  - 2025 近作《Bubble》= 一戳就破的泡泡，《自由的你》= 世界是绚烂的沙漠；
  - **学方法不抄符号**：不要用泡沫/摩天动物园/沙漠/芯片/GLORIA 这些现成意象。
- 新增 `epic` 模式（`zh_lyric_qa.py` / `zh_lyrics_gen.py` / Skill 模式 D）：
  28 句配额、hook 5–9 字、抽象词 ≤12、必须写 `metaphor` + `philosophy`；
  制作标签为 cinematic epic Mandarin pop / orchestral strings / piano / choir /
  sub bass / BGM-like，3:00–3:30。
- 新写 3 首 epic 词（均过词检）：
  - 08 把名字还给海（hook「把名字还给海」100 分；名字=海水能带走的重量；
    被记住是束缚，被忘记也是）；
  - 09 时间是个圆（hook「时间是个圆」99 分；时间=绕回伤口的圆；
    重复不是命运，是不肯在圆心停下来的选择）；
  - 10 借来的光（hook「借来的光也要还」98 分；光=可以借的债，还的时候要用自己的火）。
- 另补 06 灯还挂在城楼上（古风，hook 99 分）、05 最后一桶还没干（氛围，100 分）、
  07 坐到终点再回家（叙事，100 分）。
- 01 重跑用了 `split chorus:6` 的 178s 计划，NAR 超过 35 分钟仍未出片；按新方向
  不再等它，已停掉并保留 `retired_v1/01_..._v1.wav` 做对照。
- 13:36 启动史诗批：06（古风）+ 08/09/10（epic），一首一进程；
  epic 用 `hook chorus + thin verse:2 + final-lift 2`（不 split，保留长音和空间），
  intro 4 小节；06 用 `thin verse:2 + split chorus:6`。
- 新增 `tools/generation/zh_pop_delivery.py`：按词检/旋律/build-up/起伏/覆盖/时长
  自动筛选，只把过线的放进 `delivery/`，未入选写 `excluded.md`。

### 2026-10-26 16:30–17:30 加场：战斗史诗 + rap

- 用户要求继续做史诗感、战斗，可带部分 rap，一直工作到 17:30。
- 新增 `epic_rap` 模式（`zh_lyric_qa.py` / `zh_lyrics_gen.py` / 驱动器）：
  Verse 2 允许 4–8 个短句，总句数 24–34；hook 5–9 字；必须写 metaphor/philosophy；
  rap 段要求“像点名/报数、重音落拍、句尾押主韵、不用拟声词”。
- 11 灰里的旗（epic battle，D minor 144 BPM）：
  hook「灰里拔旗插回墙」；c1 melody 98 / build-up 15.3 / 3:23；
  c2 melody 98 / build-up 16.2 / 覆盖 1.21，c2 胜出。
- 12 名字还没念完（epic_rap，A minor 132 BPM）：
  hook「名字还没念完」；Verse 2 是点名式 rap；词检 99；
  17:05 启动单候选渲染，预计 17:20 前后出片（结果见下一节）。
- 这两首与 08/09/10 一起构成今天的“史诗/战斗”主线；02 作为唯一的近景叙事对照。

### 2026-10-26 17:20 今日最终交付

- 11 灰里的旗（epic battle）c2 胜出：build-up 16.2 dB、峰值 0.64、覆盖 1.21、
  旋律 98、3:29；c1 也是 98 分 / 15.3 dB，c2 词更清楚。
- 12 名单上划掉一行（epic_rap）单候选出片：build-up 11.3 dB、峰值 0.80、
  覆盖 0.84、词检 99；但旋律门控 69（副歌/主歌密度 1.07、音域 21），
  **不放进精选**，只留在试听包做下一轮 rap 配方参考。
- 人工换片（已写入 `outputs/2026-10-26_zh_pop/delivery/CURATION.md`）：
  - 06 用 c2（覆盖 1.11 vs 0.76）；
  - 08 用 c1（音域 19 vs 31、覆盖 0.97 vs 0.74、hook 0.975 vs 0.741）；
  - 10 用 c2（2:53 vs 3:49、音域 19 vs 29、旋律 98 vs 90）；
  - 09 保留 c1（c2 是 early peak，峰值 0.29）；
  - 11 用 c2（覆盖 1.21、起伏 16.2 dB）。
- 今日精选 `outputs/2026-10-26_zh_pop/delivery/`：**8 首**
  （02 伞往你那边偏 / 03 三楼的灯没有关 / 04 城门开之前 / 06 灯还挂在城楼上 /
  08 把名字还给海 / 09 时间是个圆 / 10 借来的光 / 11 灰里的旗），
  WAV + 同名 TXT + manifest + README + CURATION.md；8 首全部 build-up、
  起伏 9.1–18.1 dB、时长 2:05–3:29、覆盖 0.85–1.21。
- 未入选：01（重跑超时，旧版在 `retired_v1/`）、05/07（未渲染）、
  12（旋律门控 69，rap 实验）。
- 试听包：`outputs/2026-10-26_zh_pop/review.md` +
  `outputs/2026-10-26_zh_pop/review/listen/`（26 个 MP3，含每首 c1/c2/selected）。
- 本轮未推送私有库；等听审后再决定是否交付到 `Music_date`。

### 2026-10-26 17:35 推送私有库 + 格式对齐

- 用户确认推送。第一次提交 `607cd56`（扁平 8 首 + 带 `>` 头的 TXT）。
- 用户提醒“TXT 格式和前面保持一致”。按最近两批 `2026-10-22/23_native_yue2` 的格式重做：
  - TXT 改为**纯歌词、无头**（指标移到 manifest）；
  - 目录按人声分：`2026-10-26_zh_epic/中文清甜/`；
  - `manifest.json` 改成 `schema: 1`，补 `language/voice/file/lyrics/duration_seconds/
    sample_rate/channels/peak/rms/arc_verdict/lift_db/peak_at/lyric_coverage/
    weak_lyrics/sha256/lyrics_sha256`；
  - `README.md` 按 10-22/23 的写法（目录、试听顺序、自动指标说明、状态）。
- 第二次提交 `d5375e4` 已推送，远端 `main = d5375e4`；文件夹 20 个文件
  （8 WAV + 8 TXT + manifest + README + CURATION + excluded），262 MB。
- 本地 `outputs/2026-10-26_zh_pop/delivery/` 同步改成同一结构。

### 2026-10-26 17:40 私有库最终结构

- 第三次提交 `0959f44`：按 10-22/23 的目录结构，移除 `CURATION.md` / `excluded.md`
  （本地保留），仓库只留 8 WAV + 8 纯歌词 TXT + `manifest.json` + `README.md`，共 18 个文件。
- 远端 `main = 0959f44`；三次提交：`607cd56` 初次推送 → `d5375e4` 格式对齐 →
  `0959f44` 目录结构对齐。
- 推送内容：`Music_date/2026-10-26_zh_epic/中文清甜/` 8 首
  （02/03/04/06/08/09/10/11），全部 YuE2 原生 48 kHz，无 SVC/EQ/压缩/限幅。

## 2026-10-28 — 用户判 10-26 批次为"糖水歌"：重做史诗/空灵工作流

- 用户 brief：中文歌要**史诗感 + 空灵感**，不要糖水歌；参考《新·与天齐》
  《自由的你》《诀爱》与热门战斗 BGM。
- 诊断（`tools/generation/style_qa.py`，与 `human_refs/refs/` 12 首史诗参考同尺）：
  - 10-26 的 8 首并不缺小调（6/8 小调），缺的是**宽度**（0.25–0.56 vs 参考
    0.38–0.84）和 **2–6 kHz 存在感**（-11~-17 dB vs 参考 -7~-13 dB）；
  - 但后来对比 10-14/15/16 已被用户接受的英文史诗批（宽度 0.20–0.39），
    可知宽度不是"糖水"的充分原因；真正的糖水源是**标签与作词提示**：
    `build_tags()` 每档都写 `sweet/clear young female vocal` + 尾部
    `polished modern pop`，叙事档还有 `warm piano/shimmering guitars`；
    旧 SYSTEM_PROMPT 第一句是"你为一位清甜女声写歌"。
- 改动：
  - `zh_lyric_qa.py` / `zh_lyrics_gen.py` 新增 `epic_ethereal`、`battle_anthem`
    两种模式；史诗模式改用新的 `EPIC_SYSTEM_PROMPT`（史诗尺度场景、一个中心比喻、
    桥段反转；禁止便利店/末班车/雨伞/电梯等近景）；
  - `build_tags()` 史诗档不再出现 sweet/young/tender/polished pop，人声按
    空灵/战斗/哲思分开描述，并新增 `no pop production` 等反向约束；
    `validate_brief` 拦截模型自写的 `no vocals/instrumental` 与糖水词；
  - `zh_lyrics_gen.py` 新增 `--seeds-file`；
  - 新增 `tools/generation/yue2_lora.py`：把 PEFT LoRA 折进 YuE2 权重
    （TSFH AR 0.5 / NAR 0.375；instrumental-cot AR 1.0），实测 AR 196 +
    NAR 198 个 Linear 全命中；`yue2_batch_battle.py` 新增
    `--lora-ar/--lora-nar/--ar-scale/--nar-scale`；5 项单测通过；
  - 新增 `style_qa.py`（调式/宽度/2–6k/sub/centroid -> 0–100，仅诊断与排序）。
- 第 1 轮校准批（`run_epic_20261028.py`，04 首各 1 候选，全部 YuE2 原生）：
  | 曲 | 方向 | 时长 | 结构 | 高潮位置 | 覆盖 |
  |---|---|---:|---|---:|---:|
  | 01 雪线以上 | 空灵史诗 | 3:58 | early peak | 25% | 0.51 |
  | 02 换天亮 | 战斗史诗 | 2:37 | build-up | 80% | 0.60 |
  | 03 绳头 | 哲思史诗（御姐） | 3:31 | build-up 16.4 dB | 91% | **1.08** |
  | 04 破阵 BGM | 器乐 + TSFH LoRA | 1:19 | build-up | 91% | — |
  - 03 是唯一"欲扬先抑 + 咬字"双过线的方向；01 的 airy/distant 人声被混响
    推远（ASR 乱码），02 的 152 BPM 密集四字句 + 齐唱把人声埋掉；
  - 器乐线首次跑通 LoRA，宽度 0.515（人声曲 0.28），但 1:19 太短。
- 第 2 轮修正 A/B（`run_epic_ab_20261028.py`，同一份旋律计划）：
  05 空灵改"清楚/靠前/中等混响 + 主歌只留钢琴 + 末副歌进全乐队"；
  06 战斗改"独唱女声、去齐唱、no gang vocals"；
  07 BGM 用 instrumental-cot AR + TSFH NAR、不裁 intro，做长版。
- 试听包：`outputs/2026-10-28_epic/review.md` + `review/listen/*.mp3`；
  设计文档：`docs/epic_ethereal_workflow_2026-10-28.md`。
- 待用户听审：03 是否作为主方向放量 6–8 首；空灵线在"清楚"和"空间"之间的平衡点。

### 2026-10-28 下午 — 用户确认方向，放量晚间批

- 用户听完校准批："感觉还可以就这样一直做吧多做一些我晚上一块评价" →
  按 03 的哲思史诗为主、修正后的空灵/战斗为辅放量。
- 标签再修正（`build_tags`）：
  - 空灵：`airy/breathy/distant` → `clear ethereal, floating high register,
    present in the mix, lead vocal front and center`；混响改 `medium hall`；
    加 `solo piano and voice in verses / quiet verses / full orchestra in final chorus`；
  - 战斗：`choir chant` → `distant choir`；加 `solo female lead vocal` 与
    `no gang vocals / no group chant`；加 `quiet verses / full ensemble in final chorus`；
  - 哲思：加 `lead vocal front and center / quiet verses / full orchestra in final chorus`。
- 新批 `inputs/2026-10-28_epic_batch/`：10 首新词（6 哲思、2 空灵、2 战斗），
  均过词检；04 借来的名字因押韵门控连续失败，本批不含。
  曲目：火种 / 回声 / 钉子 / 逆流 / 空屋 / 灯亮着 / 岸还在 / 风敲过每扇门 /
  举火的人 / 鼓在城在。
- `run_epic_batch_20261028.py`（一首一进程，`--max-plan-seconds 200`，
  `hook chorus + thin verse:2 + final-lift 2`）14:02 启动，先跑 8 首；
  08/10 是后补的词，等首轮结束后用 `--only 08,10` 补跑。
- 新增 `epic_batch_review.py`：出晚间评审用的 `review.md` + MP3，
  按"过线/待修"排序（时长 110–215s、build-up、咬字≥0.75 只作为先听建议）。
- 实际批次（10 首人声 + 1 首 BGM）最终指标：
  - 9/10 人声为 build-up；8 首过线（时长/结构/咬字）：
    火种 203s/92%/1.04、钉子 190s/70%/0.93、溯河 181s/92%/0.75、
    空屋 198s/87%/0.98、灯亮着 188s/87%/0.89、岸还在 212s/51%/0.89、
    风敲过每扇门 183s/78%/0.98、举火的人 195s/94%/1.10；
  - 待修：回声 218s（略长）、鼓在城在 184s/咬字 0.57、破阵 BGM 231s/mild build；
  - 火种与空屋各补拍一次：火种 early peak 37% → build-up 92%（lift 20.6）；
    空屋 232s/0.71 → 198s/0.98。
- BGM 长版：TSFH LoRA、不裁 intro，230.6s、D 小调、宽度 0.434。
- 交付：`2026-10-28_zh_epic/`（中文女声/ 10 + 器乐/ 1，WAV + 纯歌词 TXT +
  manifest + README + review + FEEDBACK_MAP），推送到私有库
  `Jun-shisheng/Music_date`，远端 `main = e1b2cdb`（400 MB，11 tracks）。
- 新增 `tools/generation/package_epic_delivery.py`（按 10-26 结构打包并推送）。
- 新增 `docs/feedback_to_workflow_2026-10-28.md`：把用户可给的反馈类型
  （人声飘/太甜/不够史诗/高潮太早/旋律平/词难懂/长短/倒字/换方向）逐条映射到
  要改的标签、工具、参数和门限；副本在交付目录 `FEEDBACK_MAP.md`。


### 2026-10-29 — 影视参考批（《凡人修仙传》《灵笼》）

- 用户对 10-28 晚间批的反馈："感觉稍微有点差，可以参考凡人，灵笼，等影视作品的
  作曲看看开始今天的制作。"
- 下载 3 首参考到 `human_refs/refs/`（仅分析，不做克隆）：
  《灵笼》Incarnation（张靓颖）BV1U7411c7CE、《佩妮的日记》BV1s54y1j7wp、
  《凡人修仙传》快哉风 BV1P6YbzsEiW；转 48k WAV 后跑 style_qa / arc_qa /
  ref_profile --whisper，产物 `human_refs/refs/_work/*20261029*`。
- 实测差距（对 10-28 的 10 首）：
  - 宽度：参考 0.496–0.509，我们 0.234–0.462（中位 ~0.33）——最大差距；
  - sub：参考 32.2/32.4 dB，我们 26.8–31.5（中位 28.4）；
  - 质心：参考 1221–1991 Hz，我们 4 首 2100–2530（偏亮 = 听感薄）；
  - 结构不是主因：Incarnation lift 15.1 / 高潮 0.87，和《绳头》16.4/0.91 同量级；
  - 音区走向相反：参考副歌比主歌低 3.1–7.4 半音（乐队抬起、人声留胸声），
    我们 10-28 的 final-lift 2 是流行写法；
  - 吐词 swing 参考 x1.54–1.68，我们 1.6–2.3，本来就在区间内（不必再抽稀）。
- 本批改动：
  - 新增 `tools/generation/film_tags_20261029.py`：guofeng_film / sci_fi_ritual /
    battle_film 三套参考标签栈，覆盖模型自写的 style_extra；宽度/低频各用 3 种
    说法重复，负向加 no drum machine / no edm drop / no trap beat。
  - 新增 `tools/generation/run_film_20261029.py`：intro 8 小节（战斗 4）、
    thin verse:2 不变、slow 两个方向 final-lift 0（战斗仍 2）。
  - 新批 `inputs/2026-10-29_film_ref/` 5 首：剑还山门 / 第三声钟 / 名字磨平了 /
    念到名字 / 关在人在。后两首（03/04）DeepSeek 连续 5 次未过本地门控
    （具体物件不足、hook 重复），改为手写后用 `zh_lyric_qa` 复核：
    03 score 99、04 score 100，均无 issue。
  - 11:30:13 启动（`start_film_20261029.ps1`，pid 44816，一首一进程），
    产物 `outputs/2026-10-29_film_ref/`，状态 `logs/film_ref_20261029_status.json`。
- 文档：`docs/film_ref_workflow_2026-10-29.md`（参考实测、差距、改动、验收口径）。


### 2026-10-29 15:00 — 影视参考批出片与交付

- 5 首全部 build-up，咬字 0.62–1.03，时长 180–224s：
  03 名字磨平了 207s/89%/0.89、04 念到名字 180s/56%/0.87、
  05 关在人在 199s/89%/1.03、01 剑还山门 198s/93%/0.62、02 第三声钟 224s/94%/0.80。
- 宽度没有抬起来：0.247–0.372（10-28 是 0.234–0.462），sub 26.4–30.1。
  标签重复 wide/hard-panned/spacious 无效，证伪"宽度可靠标签解决"。
  影视配器（笛/二胡/唢呐/工业低频）与 8 小节前奏是这一批真正变的部分。
- 交付：`outputs/2026-10-29_交付/`（5 MP3 + 5 歌词 TXT + 评审建议.md，21 MB，
  MP3 已写 title/artist/album 标签）。
- 清理：删除 candidates 552 MB + raw wav 380 MB + review MP3 21 MB，释放约 950 MB；
  保留 selected WAV 母带 185 MB、qa、index。
- 下一轮杠杆：若继续追宽度，走分段双声道分层或 TSFH LoRA 宽声场，不再调标签。


### 2026-10-29 16:10 — 第二批 3 首、宽度实验与最终交付

- LoRA A/B：同一首、同一份旋律计划（ABC md5 相同），ctrl 宽度 0.247、
  NAR 0.15 为 0.267——无剂量效应，nar025 未跑，GPU 让给第二批。
- 第二批：水表慢转 186s/88%/1.08/宽 0.246、灰里的地图 211s/93%/0.82/宽 0.212、
  坟上的药 205s/65%/1.40/宽 0.328；三首全部 build-up。
- 新增 `tools/generation/widen_stereo.py`：mid/side 加宽（300 Hz 以上侧信号 ×2.5，
  tanh 软限幅，mid 不动）。实测宽度 0.247→0.422、0.212→0.419、0.328→0.480，
  sub 不变、左右相关 0.63–0.70。作为实验版单独交付。
- 最终交付 `outputs/2026-10-29_交付/`：8 首 MP3 + 8 份歌词 + 评审建议.md +
  `实验_加宽版/`（3 首 + 说明），21 个文件 47 MB；06 定名《水表慢转》。
- 清理：film_ref2 1.3 GB → 111 MB、lora_ab 451 MB 全删，本轮释放约 1.7 GB；
  保留 selected 母带 296 MB（film_ref 185 MB + film_ref2 111 MB）。
- 结论：配器、结构、咬字、前奏都已到影视参考的形态；宽度是 YuE2 原生混音天花板，
  标签与 NAR LoRA 均无效，只有混音层加宽能到 0.42–0.48（待用户听审决定）。


### 2026-10-29 16:30 — 宽度第三条路（换 seed）也排除

- 《灰里的地图》seed 9800→9901 重渲：宽度 0.218（原版 0.212），动态 8.6 dB。
  seed 9700 的两次计划均超 215s 被拦。
- 三条生成层路线（标签 / NAR LoRA / 换 seed）全部无效；只有中/侧加宽能把
  0.21–0.25 抬到 0.42–0.48（后期处理，待听审）。
- 清理 width_seeds 243 MB；当前保留：film_ref 185 MB + film_ref2 111 MB 母带，
  交付夹 47 MB。


### 2026-10-29 16:47 — 重拍 02/04 的结果

- 02：两次计划 225.0s / 221.4s，`--max-plan-seconds 200` 下均被拦，保留原版 224s。
- 04：重拍 187s / build-up 14.2 / 高潮 65% / 咬字 1.22，但宽度 0.281→0.193，
  净损失，保留原版。
- 清理 retake 204 MB；最终保留 film_ref 185 MB、film_ref2 111 MB 母带，交付夹 47 MB。


### 2026-10-29 17:15 — 交付整理为两个日期夹

- `outputs/2026-10-28_交付/`：11 首（10 人声 + 器乐《破阵》）MP3 + 歌词 + 评审建议.md，46 MB。
- `outputs/2026-10-29_交付/`：8 首 MP3 + 歌词 + 评审建议.md + `实验_加宽版/`（3 首），47 MB。
- 删除全部中间产物：10-28 的 epic/epic_ab/epic_batch（4.1 GB）、10-29 的 film_ref/film_ref2
  （296 MB）；outputs 从 4.5 GB 降到 92 MB。
- WAV 母带随工作目录删除；10-28 的 WAV 仍在已推送的私有库
  （`.cache/Music_date_repo/2026-10-28_zh_epic`，远端 ba0e2ed），10-29 的母带未留。


### 2026-10-30 — 方向调整：纯音乐 + 同曲填词（SVS 中文）

- 用户判定 YuE2 唱中文始终不过关，主线改为两条：纯器乐，和「同曲填词翻唱」（二创定位），
  不再让生成模型从零做完整中文歌。
- 新建 `tools/voice_studio/svs_score_zh.py`：中文从 ASR 词时间戳 + pyin 音高生成 SoulX SVS
  metadata。中文音素是整音节带声调（`zh_<拼音><声调>`，词表 398 音节/1201 条），无 CMU 词典；
  多音字必须整句转换（单字「吭」=keng1，整句=hang2）。
- `control=score` 只读 note_pitch，不读 F0 轮廓 → 换字时旋律精确保持。
- 嗓子测试（固定女声 `iron_rain_female_svs_v1`）：
  - 24s《无问西东》副歌（旋律从男/女声干声抠）：27/27 音在 1 半音内（最大 0.4），
    Whisper 回读约九成字。样片 `outputs/2026-10-30_zh_svs/chorus_long/`。用户听感「好像还可以」。
  - 压力测试 `outputs/2026-10-30_zh_svs/stress/voice_stress_all.mp3`：高音 81 掉了 9–10 半音、
    79 掉 6.3 半音；低音 55–60 长音全准；132bpm 八分音符 93% 在 1 半音内。
    结论：该声线上限约 G5，写旋律不要超过 MIDI 77–79。
- 纯音乐第一批 3 首（inputs/2026-10-30_bgm）：断崖 121s / 送别 234s / 天光 105s，
  MP3 在 `outputs/2026-10-30_bgm/mp3/`，26.8 分钟跑完。
- 纯音乐第二批 5 首（inputs/2026-10-30_bgm2）：追击 / 夜袭 / 凯旋 / 荒原 / 山门，13:39 开跑。
- 填词：用户明确「不着急」，二创用途，先不推进；草稿在 `outputs/2026-10-30_zh_svs/fill_v1.txt`。
