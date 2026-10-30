"""Write Mandarin pop briefs for YuE2, one title at a time, with a local gate.

Why this exists (2026-10-26): the 2026-10-22/23 Chinese batch used generic
Mandarin-pop briefs -- every song had the same register (星光/远方/梦想/世界),
the chorus was a full sentence rather than a hook, and the lyric did not tell
the melody where to breathe.  The user's brief for this week is lyrics + tune
for Chinese songs, with 悬溺 and 新·与天齐 named as references.

This script is the front half of a Chinese YuE2 run:

  * a *direction* fixes the scene, the addressee, the secret, the image bank,
    the key and the BPM, so four titles cannot converge;
  * DeepSeek writes the actual lyric, hook, title and production descriptors;
  * the local gate (`zh_lyric_qa.py`) rejects the brief before it reaches the
    GPU: wrong section tags, no hook, cliche/AI-copywriter language,
    onomatopoeia/humming (explicitly forbidden by the user), wrong line budget,
    abstract-word overload, no concrete objects, 3rd-tone pileups.

The generator never reuses an earlier title's hook or opening image: the used
list is fed back into every later prompt.

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/zh_lyrics_gen.py --out-dir inputs/2026-10-26_zh_pop --count 4
  $P -I tools/generation/zh_lyrics_gen.py --out-dir inputs/2026-10-26_zh_pop --rebuild-tags

Rerun-safe: an existing title is skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT / "tools" / "generation"))
from zh_lyric_qa import MODES, audit  # noqa: E402

SECTION_ORDER = [
    "Intro", "Verse 1", "Pre-Chorus 1", "Chorus", "Verse 2",
    "Pre-Chorus 2", "Chorus 2", "Bridge", "Final Chorus", "Outro",
]

# Four directions for the 2026-10-26 pilot.  Deliberately different modes:
# two narrative, one atmospheric, one guofeng.  The 2026-10-22 set used the same
# register for all sixteen Chinese titles; that is the failure this fixes.
SEEDS: list[dict] = [
    {
        "slug": "01_last_bus_convenience_store",
        "mode": "narrative",
        "angle": "末班车开走以后，两个人在便利店门口分一盒关东煮；谁都没有说破，但都在等对方先开口",
        "images": "关东煮的纸盒、塑料叉、便利店的白灯、末班车的尾灯、关东煮的热气、玻璃门上的雨、找零的硬币",
        "addressee": "一个认识了很久、快要错过的人",
        "secret": "其实车票早就买好了两张，只是一直没拿出来",
        "key": "C major", "bpm": 96, "voice": "sweet clear young female vocal, close and intimate",
        "rhyme": "an/ian/uan", "title_hint": "末班车以后",
    },
    {
        "slug": "02_umbrella_on_the_wrong_side",
        "mode": "narrative",
        "angle": "雨天共撑一把伞，伞一直往对方那边偏；走到家门口才发现自己半边肩膀全湿了",
        "images": "伞骨、湿掉的肩膀、公交站牌、积水里的霓虹、便利店塑料袋、袖口的水、楼道声控灯",
        "addressee": "一个习惯了照顾别人、不太会说自己冷的人",
        "secret": "那把伞其实是对方上次落下的，一直没还",
        "key": "G major", "bpm": 104, "voice": "sweet clear young female vocal, bright and tender",
        "rhyme": "ang/iang/uang", "title_hint": "伞往你那边偏",
    },
    {
        "slug": "03_elevator_at_three",
        "mode": "atmospheric",
        "angle": "凌晨三点，电梯停在错误的楼层；门开了一次，走廊里没有人，只有一户还亮着灯",
        "images": "电梯按钮、走廊声控灯、防盗门、手机录音的红点、冰箱的嗡鸣、没喝完的水、猫眼",
        "addressee": "一个已经搬走、但还留着钥匙的人",
        "secret": "那段没发出去的语音，其实已经录好了",
        "key": "A minor", "bpm": 72, "voice": "sweet clear young female vocal, breathy low register, restrained",
        "rhyme": "an/ian/uan", "title_hint": "三楼的灯没有关",
        "hook_hint": "三楼的灯没有关",
    },
    {
        "slug": "04_before_the_gate_opens",
        "mode": "guofeng",
        "angle": "城门开之前，一个人站在雪里，把名字和来路都放下；不求仙，不求神，只求这一剑由自己递出",
        "images": "城门、雪、剑鞘、马蹄印、火把、旧甲、酒碗、城墙上的霜",
        "addressee": "一个被天命安排过、但决定自己走的人",
        "secret": "他等的不是援军，是城门开的那一声",
        "key": "D minor", "bpm": 150, "voice": "clear young female vocal with power, guofeng pop",
        "rhyme": "an/ian/uan", "title_hint": "城门开之前",
        "hook_hint": "我自己的路自己开",
    },
    {
        "slug": "05_last_load_laundromat",
        "mode": "atmospheric",
        "angle": "深夜自助洗衣店，最后一桶衣服还在转；等一个说“明天来拿”的人，洗衣机停下时整间店只剩嗡嗡声",
        "images": "洗衣机圆窗、泡沫、烘干机、塑料椅、洗衣液、硬币、湿衣服、日光灯、门口的伞、烟头",
        "addressee": "一个说会回来拿衣服、但一直没有来的人",
        "secret": "那桶衣服其实是对方的，故意没有烘干，好让ta再来一次",
        "key": "E minor", "bpm": 70,
        "voice": "sweet clear young female vocal, breathy low register, restrained",
        "rhyme": "ang/iang/uang", "title_hint": "最后一桶还没干",
        "hook_hint": "最后一桶还没干",
    },
    {
        "slug": "06_lantern_on_the_pass",
        "mode": "guofeng",
        "angle": "边关最后一盏灯，守灯人等一封不会来的家书；雪封了路，他把灯挂在城楼上，让回来的人能看见",
        "images": "灯笼、城楼、雪、家书、火折子、马蹄、关隘、旧棉衣、铜铃、酒",
        "addressee": "一个去了关外、说好春天回来的人",
        "secret": "那封信其实早就到了，被他压在灯座下面，怕看完就真的断了",
        "key": "A minor", "bpm": 140,
        "voice": "clear young female vocal with power, guofeng pop",
        "rhyme": "ang/iang/uang", "title_hint": "灯还挂在城楼上",
        "hook_hint": "灯还挂在城楼上",
    },
    {
        "slug": "07_ride_to_the_last_stop",
        "mode": "narrative",
        "angle": "地铁坐过站，索性坐到终点；终点站只剩便利店和一张长椅，手机里那条没发出去的消息还停在输入框",
        "images": "地铁报站、扶手、耳机、终点站、便利店、矿泉水、未发送消息、站台长椅、末班车、玻璃门",
        "addressee": "一个一直在等回复的人",
        "secret": "那条消息写的是“我到了”，其实那天根本没有去",
        "key": "F major", "bpm": 100,
        "voice": "sweet clear young female vocal, bright and tender",
        "rhyme": "an/ian/uan", "title_hint": "坐到终点再回家",
        "hook_hint": "坐到终点再回家",
    },
    {
        "slug": "08_give_the_name_back_to_the_sea",
        "mode": "epic",
        "angle": "一个人在海边把自己的名字还给海，从此没有人能叫住他；他以为自由了，后来发现被忘记也是一种重量",
        "images": "海、名字、沙滩、潮水、灯塔、盐、空瓶、旧照片、回声、岸",
        "addressee": "一个曾经叫过他名字的人",
        "secret": "他其实每年都会回到同一片海滩，听有没有人叫他",
        "key": "E minor", "bpm": 84,
        "voice": "clear young female vocal with power, cinematic",
        "rhyme": "ai/uai", "title_hint": "把名字还给海",
        "hook_hint": "把名字还给海",
        "metaphor": "名字是海水能带走的重量",
        "philosophy": "被记住是束缚，被忘记也是；自由不是没有名字，而是敢再给自己取一个",
    },
    {
        "slug": "09_time_is_a_circle",
        "mode": "epic",
        "angle": "我们以为在往前走，其实一直在绕回同一个伤口；直到有一天，我们选择在圆心停下来，不再逃",
        "images": "钟、指针、圆规、伤口、走廊、回声、沙漏、轨道、门、光",
        "addressee": "一个反复回到同一天的人",
        "secret": "那个伤口其实是他自己每年同一天重新划开的",
        "key": "A minor", "bpm": 92,
        "voice": "clear young female vocal with power, cinematic",
        "rhyme": "an/ian/uan", "title_hint": "时间是个圆",
        "hook_hint": "时间是个圆",
        "metaphor": "时间不是直线，是一个绕回伤口的圆",
        "philosophy": "重复不是命运，是我们不肯在圆心停下来的选择",
    },
    {
        "slug": "10_borrowed_light",
        "mode": "epic",
        "angle": "我们靠别人留下的光活着；有一天光灭了，才发现自己也能发光",
        "images": "灯、火柴、镜子、月亮、借条、余烬、窗、海面、手电、星",
        "addressee": "一个已经离开、但把光留下的人",
        "secret": "那盏灯其实早就没油了，是他自己一直在擦",
        "key": "D minor", "bpm": 88,
        "voice": "clear young female vocal with power, cinematic",
        "rhyme": "an/ian/uan", "title_hint": "借来的光",
        "hook_hint": "借来的光也要还",
        "metaphor": "光是可以借的债，还的时候要用自己的火",
        "philosophy": "我们以为自己在借光，其实是在学怎么成为光",
    },
    {
        "slug": "11_banner_in_the_ash",
        "mode": "epic",
        "angle": "一场已经输掉的战役，最后一个人把旗从灰里拔出来，插在没人守的城墙上；不是为了赢，是为了让后来的人知道这里有人站过",
        "images": "旗、灰、断矛、城墙、鼓、火、雪、名字、钟、乌鸦",
        "addressee": "一个已经倒下的同伴",
        "secret": "旗其实早就烧掉一半，他举的是剩下的那一半",
        "key": "D minor", "bpm": 144,
        "voice": "clear young female vocal with power, cinematic battle",
        "rhyme": "ang/iang/uang", "title_hint": "灰里的旗",
        "hook_hint": "把旗插回城上",
        "metaphor": "旗不是布，是‘还有人站着’这件事",
        "philosophy": "输赢会过去，但有人站过这件事不会",
    },
    {
        "slug": "12_names_not_finished",
        "mode": "epic_rap",
        "angle": "战后的点名，一个人念着阵亡者的名字；念到一半发现自己的名字也在名单上，于是把自己那行划掉，继续念别人",
        "images": "名单、铅笔、灯、雨、空盔甲、钟、名字、纸、火、门",
        "addressee": "一个不在名单上的人",
        "secret": "他把自己那一行划掉了，因为“还没轮到”",
        "key": "A minor", "bpm": 132,
        "voice": "clear young female vocal with power, rap flow in verse 2",
        "rhyme": "an/ian/uan", "title_hint": "名字还没念完",
        "hook_hint": "名字还没念完",
        "metaphor": "点名是把死人留在名册里，也是把活人留在世上",
        "philosophy": "被念到名字的人没有消失；活着的人要替他们念完",
        "rap": True,
    },
]

MODE_BRIEFS = {
    "narrative": {
        "name": "近景叙事流行",
        "section_budget": {
            "Intro": 1, "Verse 1": 4, "Pre-Chorus 1": 2, "Chorus": 4,
            "Verse 2": 4, "Pre-Chorus 2": 2, "Chorus 2": 4, "Bridge": 2,
            "Final Chorus": 4, "Outro": 1,
        },
        "total": 28,
        "register": "主歌低、近、像说话；预副歌短句加速；副歌打开但不要喊；末副歌比首副歌高一点。",
        "hook": "副歌第一句就是 hook，5–8 个汉字，能脱离旋律单独说。",
    },
    "atmospheric": {
        "name": "暗色氛围流行（悬溺型，但不许拟声/哼唱）",
        "section_budget": {
            "Intro": 1, "Verse 1": 3, "Pre-Chorus 1": 1, "Chorus": 3,
            "Verse 2": 3, "Pre-Chorus 2": 1, "Chorus 2": 3, "Bridge": 1,
            "Final Chorus": 3, "Outro": 1,
        },
        "total": 20,
        "register": "整体低、慢、留白多；副歌不要靠喊，靠气声和空间；句尾长音优先开口音。",
        "hook": "副歌第一句是 hook，3–8 个真词；不要拟声词、不要哼唱、不要英文。",
    },
    "guofeng": {
        "name": "古风燃向（新·与天齐型）",
        "section_budget": {
            "Intro": 1, "Verse 1": 4, "Pre-Chorus 1": 2, "Chorus": 4,
            "Verse 2": 4, "Pre-Chorus 2": 2, "Chorus 2": 4, "Bridge": 2,
            "Final Chorus": 4, "Outro": 1,
        },
        "total": 28,
        "register": "主歌压住、像在雪里说话；预副歌一句比一句短；副歌喊口号，但每个口号落在动作/物件上；末副歌升调。",
        "hook": "副歌第一句是 5–8 字口号，四三或五二切分，落在强拍，能单独喊；不要无词。",
    },
    "epic": {
        "name": "史诗感 / BGM / 思辨中文曲（邓紫棋式中心比喻）",
        "section_budget": {
            "Intro": 1, "Verse 1": 4, "Pre-Chorus 1": 2, "Chorus": 4,
            "Verse 2": 4, "Pre-Chorus 2": 2, "Chorus 2": 4, "Bridge": 2,
            "Final Chorus": 4, "Outro": 1,
        },
        "total": 28,
        "register": "主歌低、克制、像旁白；预副歌把意象收紧；副歌打开成电影感的大线条；桥段必须有一次哲学反转；末副歌最高最宽。",
        "hook": "副歌第一句是 5–9 个字的直白句，能单独说出口；它必须和中心比喻是同一个东西。",
    },
    "epic_rap": {
        "name": "史诗战斗 + 一段 rap（Verse 2）",
        "section_budget": {
            "Intro": 1, "Verse 1": 4, "Pre-Chorus 1": 2, "Chorus": 4,
            "Verse 2": 6, "Pre-Chorus 2": 2, "Chorus 2": 4, "Bridge": 2,
            "Final Chorus": 4, "Outro": 1,
        },
        "total": 30,
        "register": "主歌低、克制；Verse 2 是 rap，短句、重音、像点名；副歌是电影感大线条；桥段反转；末副歌最高最宽。",
        "hook": "副歌第一句是 5–9 个字的直白句，能单独说出口；它必须和中心比喻是同一个东西。",
    },
    "epic_ethereal": {
        "name": "空灵史诗（诀爱型：极简、辽阔、宿命感）",
        "section_budget": {
            "Intro": 1, "Verse 1": 4, "Pre-Chorus 1": 2, "Chorus": 4,
            "Verse 2": 4, "Pre-Chorus 2": 2, "Chorus 2": 4, "Bridge": 2,
            "Final Chorus": 4, "Outro": 1,
        },
        "total": 28,
        "register": "主歌低、慢、几乎耳语，只给画面不给结论；预副歌一句比一句短，最后一句悬在空里；"
                    "副歌不是喊，是把声音放到很远的地方，长音收句；桥段掉下去，只剩一个字或一件物；"
                    "末副歌才让整个乐队和合唱进来。",
        "hook": "副歌第一句 5–10 字，能单独说；它必须是一个画面或一个动作，不许是抽象口号。",
    },
    "battle_anthem": {
        "name": "战斗史诗（新·与天齐型：口号、强重音、逆天）",
        "section_budget": {
            "Intro": 1, "Verse 1": 4, "Pre-Chorus 1": 2, "Chorus": 4,
            "Verse 2": 4, "Pre-Chorus 2": 2, "Chorus 2": 4, "Bridge": 2,
            "Final Chorus": 4, "Outro": 1,
        },
        "total": 28,
        "register": "主歌压低、像在风里说话；预副歌短句加速，一句一顿；副歌是口号，四三切分落在强拍；"
                    "桥段把口号背后的代价说出来；末副歌升调，喊完收住。",
        "hook": "副歌第一句是 5–9 字口号，四三或五二切分，能单独喊；每个字都要有动作或物件，不要空喊宿命。",
    },
}

SYSTEM_PROMPT = """你是当代华语流行歌曲的作词人，不是诗人，也不是励志文案写手。
你为一位清甜女声写歌，目标是 2024–2026 华语热歌的听感：具体、口语、有一个能被记住的 hook。

写之前先记住这些硬规则（违反任何一条都会被本地门控打回）：

1. 一首歌只讲一个场景、一个对象、一个秘密。对象用“你”。
2. 主歌写近景：时间、地点、一个动作、一个物件。不要解释情绪，让物件承担情绪。
3. 副歌第一句就是 hook，能脱离旋律单独说；hook 在全曲出现 2–3 次，不能更多。
4. 预副歌用短句排比加速，最后一句悬住，不要落回结论。
5. 末副歌换一半新词，只把 hook 留回来；开头和结尾不要复用中间的原句。
6. 桥段必须给新信息（一个动作、一个原因、一个反转），2–3 句，低音区。
7. 押实词，不押空词。句尾优先具体名词/动词，不要落在“的/了/着/吗/呢”。
8. 声调服从旋律：长音/最高音优先一声、二声、开口音（a/an/ang/ao/ai）；
   三声不放在最高最长音；一句内三声连读不超过 2 个；虚词不占句尾长音。
9. 句长：主歌 7–11 字，副歌 5–9 字，一句一口气。不要长定语套长定语，不要文言倒装。
10. 每首至少 4 个可触摸的物件；抽象大词（星光/远方/梦想/世界/温柔/拥抱/
    眼泪/孤独/永恒/翅膀/岁月/时光/心跳/命运/守护/希望/未来/勇敢/奇迹/宇宙）
    全曲合计 ≤ 4 次，且不能进 hook。
11. 严禁拟声词、哼唱、无词吟唱：不要“啊——”“呀咿呀”“啦啦啦”“na na na”，
    也不要任何只有语气词的一行。用户明确要求避免。
12. 不要英文单词、不要舞台提示、不要括号、不要任何解释性文字混进歌词。
13. 不要用 AI 文案腔：不负韶华、奔赴山海、万物、治愈、温柔以待、闪闪发光、
    一路生花、未来可期、心之所向、眼里有光、热爱可抵、人间值得、乘风破浪、
    披荆斩棘、向阳而生、顶峰相见、星河滚烫。这些一律禁止。
14. 不要抄任何现有歌词；只学结构和手法，不学具体句子。

只输出严格 JSON，不要 markdown，键如下：
  title       : 3–6 个汉字的中文标题
  hook        : 副歌里重复的那一句，5–8 个汉字（古风可 5–8，氛围可 3–8）
  lyrics      : 完整歌词，带段落标签，换行分隔
  style_extra : 8–14 个英文制作描述（乐器、质感、编曲走向），不要描述人声和段落动态
  notes       : 2–4 句中文，说明这首歌的场景、对象、秘密，以及你为什么这样写 hook
  metaphor    : 全曲的中心比喻（一句；epic 模式必填）
  philosophy  : 这首歌的思辨/哲学命题（一句；epic 模式必填）
"""

# 2026-10-28: the 10-26 batch was rejected as 糖水歌.  The old base prompt
# ("你为一位清甜女声写歌…具体、口语") is exactly what produced it, so the
# epic modes get their own system prompt instead of inheriting the pop one.
EPIC_SYSTEM_PROMPT = """你是中文史诗 / 电影感歌曲的作词人，不是流行情歌写手，也不是励志文案写手。
你写的歌要能放进电影预告、战斗 BGM、游戏片尾：有宿命重量，有辽阔空间，但每个字都要听得清。

先记住这三种参照，只学方法，不抄句子：
- 邓紫棋：一首歌只用一个中心比喻，hook 直白到能单独说出口，桥段把这个比喻翻过来；
- 《新·与天齐》：口号句四三切分（4 字 + 3 字），落在强拍，能单独喊；
- 《诀爱》：词要少，画面要大；人声退后，空间上前；不解释情绪，只呈现画面。

硬规则（违反任何一条都会被本地门控打回）：

1. 禁止日常近景。不要便利店、末班车、雨伞、电梯、外卖、微信、手机、地铁、出租屋、
   校服、奶茶。主歌的场景必须是史诗尺度里的具体动作：雪线、山口、城墙、河床、
   盐地、断桥、钟楼、战场、崖边、荒原。场景大，动作必须小（一个人蹲下、点灯、拴绳、埋东西）。
2. 全曲只有一个中心比喻，必须贯穿主歌、副歌、桥段，并且就是歌名 / hook。
   比喻要能用一个具体物件说清楚（灯、绳、盐、钟、旗、岸、雪、门、钉子、镜子）。
3. 副歌第一句就是 hook，5–9 个汉字，直白，能脱离旋律单独说；hook 最后一个字必须是
   开口音（a/an/ang/ao/ai 韵母），不要用 i/ü/e 收尾。
4. 主歌低、慢、克制：只给画面和动作，不解释情绪，不下结论。
5. 预副歌 2–3 句，短句排比，一句比一句短，最后一句悬住，不要落回结论。
6. 副歌是"把比喻打开"，不是喊口号。长音落在开口音上；不要每句都升维。
7. 桥段必须把比喻翻过来：先承认它，再反转它（悖论 / 反问 / 代价）。2–3 句，低音区，可换韵。
8. 末副歌换一半新词，只把 hook 留回来；最高、最长的音留在这里的开口音字上。
9. 押实词，不押空词。句尾优先具体名词 / 动词；段内至少 70% 的句尾落在同一韵组；
   句尾不要落在"的/了/着/吗/呢/吧/啊"。
10. 句长：主歌 7–11 字，副歌 5–9 字，一句一口气。不要长定语套长定语，不要文言倒装。
11. 每首至少 4 个可触摸的物件 / 动作。抽象大词（星光/远方/梦想/世界/永恒/宿命/荣耀/
    深渊/救赎/信仰/希望/未来/勇敢/奇迹/宇宙/灵魂/命运）全曲合计 ≤ 12 次，且不能进 hook；
    每个抽象词必须落在一个具体动作或物件上。
12. 严禁拟声词、哼唱、无词吟唱；严禁英文单词、舞台提示、括号、解释性文字。
13. 严禁 AI 文案腔：不负韶华、奔赴山海、万物、治愈、温柔以待、闪闪发光、一路生花、
    未来可期、心之所向、眼里有光、热爱可抵、人间值得、乘风破浪、披荆斩棘、向阳而生、
    顶峰相见、星河滚烫、逆天改命（作为原句直接抄）。
14. 不要抄任何现有歌词，不要复用"泡沫/沙漠/摩天动物园/芯片/GLORIA"这类现成符号。

只输出严格 JSON，不要 markdown，键如下：
  title       : 3–6 个汉字的中文标题（必须是中心比喻本身）
  hook        : 副歌里重复的那一句，5–9 个汉字，开口音收尾
  lyrics      : 完整歌词，带段落标签，换行分隔
  style_extra : 8–14 个英文制作描述（乐器、质感、编曲走向），不要描述人声和段落动态
  notes       : 2–4 句中文，说明中心比喻、桥段的反转、以及 hook 为什么能单独说
  metaphor    : 全曲的中心比喻（一句）
  philosophy  : 这首歌的思辨 / 哲学命题（一句）
"""

EPIC_MODES = {"epic", "epic_rap", "epic_ethereal", "battle_anthem"}


def system_prompt_for(mode: str) -> str:
    return EPIC_SYSTEM_PROMPT if mode in EPIC_MODES else SYSTEM_PROMPT


def _build_user_prompt(seed: dict, used: list[dict]) -> str:
    mode = seed["mode"]
    brief = MODE_BRIEFS[mode]
    budget = "\n".join(f"  [{label}] {n} 行" for label, n in brief["section_budget"].items())
    parts = [
        f"模式：{brief['name']}",
        f"选题：{seed['angle']}",
        f"可触摸的物件（必须用，不要换成抽象词）：{seed['images']}",
        f"对象：{seed['addressee']}",
        f"秘密（桥段才揭示）：{seed['secret']}",
        f"音乐框架：{seed['key']}，{seed['bpm']} BPM。",
        f"主韵：{seed['rhyme']}。每一段内部至少 70% 的句尾要落在同一个韵组上；"
        f"主歌、副歌可以用不同韵，但段内必须统一；桥段可换韵。"
        f"句尾不要落在“的/了/着/吗/呢/吧/啊”。",
        f"标题方向：{seed['title_hint']}（可以改，但必须是具体意象，不要抽象大词）。",
        *([f"Hook 方向：{seed['hook_hint']}（可以改，但必须保留具体意象和开口音收尾）。"]
          if seed.get("hook_hint") else []),
        f"人声：{seed['voice']}。",
        f"段落与句数（必须逐段命中）：\n{budget}\n 总句数 {brief['total']}。",
        f"段落写法：{brief['register']}",
        f"Hook：{brief['hook']}",
        "hook 最后一个字必须是开口音（a/an/ang/ao/ai 韵母），"
        "古风可用“天/前/关/开/山/还/寒/远”收尾；不要用 i/ü/e 收尾。",
        "句内三声连读最多 3 个（含 3 个），超了就换字；"
        "长音/最高音优先一声、二声、开口音。",
        "除了 hook，任何一句最多出现 2 次；副歌 2 不要整段复制副歌 1，"
        "至少换 2 句新词；末副歌也只留 hook，其余换新。",
        "写之前先数一遍每段句数，再数一遍 hook 出现次数，再检查一遍段内韵脚。",
        *([(
            "史诗/BGM 模式额外要求：\n"
            "A. 全曲只用一个中心比喻，必须贯穿主歌、副歌、桥段；不要中途换成另一个比喻。\n"
            "B. 三段推进：主歌写这个比喻的字面场景；预副歌收紧；副歌把比喻说成一句直白的话；\n"
            "   桥段必须把这个比喻翻过来看一次（反转/悖论）；末副歌给出哲学结论。\n"
            "C. 对句和行中停顿：多用 4+4 或 5+5 的停顿，行尾和停顿处尽量押韵；\n"
            "   用对偶、悖论、反问制造思辨感，但不要写成论文。\n"
            "D. hook 必须直白、能单独说出口，和中心比喻是同一个东西；不要用生僻词。\n"
            "E. 允许哲学抽象词，但每个抽象词必须落在一个具体动作/物件上；\n"
            "   全曲抽象大词不超过 12 次，hook 里不能有。\n"
            "F. 不要抄邓紫棋的具体符号（泡沫、摩天动物园、沙漠、芯片、GLORIA）；\n"
            "   只学她的‘一个中心比喻 + 直白 hook + 哲学反转’方法。\n"
            "G. 画面要有电影感/BGM 感：大空间、单一光源、缓慢推进；不要日常品牌和网络梗。\n"
            "H. 在 JSON 里写出 metaphor（中心比喻）和 philosophy（思辨命题），各一句。\n"
            f"中心比喻方向：{seed.get('metaphor', '')}\n"
            f"思辨命题方向：{seed.get('philosophy', '')}"
        )] if mode in EPIC_MODES else []),
        *([(
            "Rap 段要求（Verse 2）：\n"
            "6–8 个短句，每句 4–8 个汉字，重音落在拍上，像点名/报数一样往前推；\n"
            "不要写成朗诵，也不要拟声词；押韵可以密，但每句必须有具体信息。\n"
            "副歌仍然是唱，不要整首 rap。"
        )] if seed.get("rap") else []),
        *([(
            "战斗史诗（battle_anthem）额外要求：\n"
            "A. 副歌只有第一句是 hook；Chorus / Chorus 2 / Final Chorus 的其余句子必须是新词，"
            "不许把 hook 换个字再唱；hook 全曲只出现 3 次。\n"
            "B. 口号句按 4+3 切分，重音落在第 1、4、7 个字上；全曲最多 2 个口号句。\n"
            "C. 战斗画面要具体：火把、城门、鼓、断箭、雪、铁、绳梯、号角；"
            "不要空喊宿命/荣耀/深渊。\n"
            "D. 每一句最多 3 个三声字；写完整首后逐句数一遍。"
        )] if mode == "battle_anthem" else []),
    ]
    if used:
        parts.append(
            "本轮已经用过的标题/hook/开场画面，禁止复用：\n"
            + "\n".join(
                f"- {u['title']}：hook「{u['hook']}」；开场「{u.get('first_line', '')}」"
                for u in used
            )
        )
    parts.append("现在写这一首。")
    return "\n\n".join(parts)


def _read_api_key(env_path: Path) -> tuple[str, str]:
    key = ""
    base = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip()
            elif line.startswith("DEEPSEEK_BASE_URL="):
                base = line.split("=", 1)[1].strip()
    return key, (base or "https://api.deepseek.com")


def request_brief(client, base_url: str, key: str, seed: dict,
                  used: list[dict], problems: list[str]) -> dict:
    import httpx

    user = _build_user_prompt(seed, used)
    if problems:
        user += ("\n\n上一版被本地门控打回了，必须逐条修正：\n"
                 + "\n".join(f"- {p}" for p in problems))
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt_for(seed["mode"])},
            {"role": "user", "content": user},
        ],
        "temperature": 0.95,
        "response_format": {"type": "json_object"},
    }
    last = None
    for attempt in range(3):
        try:
            resp = client.post(f"{base_url}/v1/chat/completions", json=payload,
                               headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"})
            if resp.status_code != 200:
                last = RuntimeError(f"deepseek HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(2 + attempt * 3)
                continue
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(match.group(0) if match else content)
        except httpx.TransportError as exc:
            last = exc
            time.sleep(2 + attempt * 3)
    raise last or RuntimeError("deepseek: no response after retries")


_ASCII_WORD = re.compile(r"[A-Za-z]{2,}")
_PARENS = re.compile(r"[()（）]")
_ILLEGAL_SECTION = re.compile(r"^\[([^\]]+)\]\s*$", re.MULTILINE)


def validate_brief(data: dict, seed: dict) -> list[str]:
    problems: list[str] = []
    for key in ("title", "hook", "lyrics", "style_extra", "notes"):
        if not str(data.get(key, "")).strip():
            problems.append(f"缺少 {key}")
    if seed["mode"] in EPIC_MODES:
        for key in ("metaphor", "philosophy"):
            if not str(data.get(key, "")).strip():
                problems.append(f"epic 模式缺少 {key}（中心比喻/思辨命题）")

    lyrics = str(data.get("lyrics", "")).strip()
    tags = _ILLEGAL_SECTION.findall(lyrics)
    if tags != SECTION_ORDER:
        problems.append(f"段落标签是 {tags}，必须是 {SECTION_ORDER}")
    sung = "\n".join(line for line in lyrics.splitlines()
                     if line.strip() and not line.strip().startswith("["))
    if _ASCII_WORD.search(sung):
        problems.append("歌词里出现了英文单词")
    if _PARENS.search(sung):
        problems.append("歌词里出现了括号/舞台提示")

    report = audit(lyrics, title=str(data.get("title", "")), mode=seed["mode"])
    problems.extend(report.issues)
    if report.hook and not report.hook_open_vowel:
        problems.append(
            f"hook {report.hook!r} 最后一个字不是开口音；"
            "改成 a/an/ang/ao/ai 韵母收尾，最高音才唱得住")

    hook = str(data.get("hook", "")).strip()
    if hook:
        if hook not in lyrics:
            problems.append(f"声明的 hook {hook!r} 没有出现在歌词里")
        elif lyrics.count(hook) > 3:
            problems.append(f"hook {hook!r} 出现 {lyrics.count(hook)} 次（需≤3）")

    style = str(data.get("style_extra", ""))
    if "male vocal" in style.lower():
        problems.append("style_extra 里出现了 male vocal（本周是女声）")
    if re.search(r"[㐀-鿿]", style):
        problems.append("style_extra 必须是英文制作描述，不能有中文")
    if seed["mode"] in EPIC_MODES:
        # 2026-10-28: the 03 model wrote "no vocals" into its own production
        # notes, which would fight the singer the whole pipeline exists for.
        if re.search(r"\b(no vocals?|no singing|without vocals?|instrumental)\b",
                     style, re.IGNORECASE):
            problems.append("epic 模式是带人声的，style_extra 不能写 no vocals/instrumental")
        if re.search(r"\b(sweet|tender|intimate|young female|polished pop)\b",
                     style, re.IGNORECASE):
            problems.append("style_extra 里出现了糖水流行词（sweet/tender/intimate/young female/polished pop）")
    return problems


def build_tags(seed: dict, style_extra: str) -> str:
    """Deterministic tag head + the model's own production descriptors.

    2026-10-28: the epic modes no longer share the pop tail.  "polished modern
    pop" plus "sweet/young/tender" was the sugar-water recipe the user rejected;
    the epic heads speak in orchestra/choir/space and the tail forbids the pop
    register instead of asking for it.
    """
    mode = seed["mode"]
    bpm, key = seed["bpm"], seed["key"]
    if mode == "epic_ethereal":
        # 2026-10-28 round 1: `airy and breathy, distant and vast` pushed the
        # vocal into the reverb (ASR coverage 0.51, transcript was noise).
        # Round 2 (05) moved it to clear/present and coverage rose to 0.79.
        head = [
            "clear ethereal female vocal, floating high register, present in the mix, "
            "pure tone, restrained vibrato, lead vocal front and center",
            "cinematic epic Mandarin ballad, film score",
            f"{bpm} bpm", key, "sweeping orchestral strings", "low cello drone",
            "grand piano", "angelic choir", "timpani", "deep sub bass",
            "medium hall reverb", "wide cinematic stereo image", "sparse percussion",
            "solo piano and voice in verses", "quiet verses",
            "full orchestra enters in final chorus", "slow build",
        ]
        if bpm >= 88:
            head += ["soaring powerful climax"]
    elif mode == "battle_anthem":
        # Round 1: `choir chant` + 152 BPM dense lines buried the lead vocal
        # (coverage 0.60).  Round 2 (06) solo/front + no gang vocals lifted it
        # to 1.20; the distant choir stays as production, not as the singer.
        head = [
            "solo female lead vocal, powerful chest-mix belt, clear diction, "
            "front and center",
            "epic Chinese battle anthem, guofeng battle music",
            f"{bpm} bpm", key, "taiko war drums", "erhu", "dizi", "guzheng",
            "orchestral strings", "brass fanfare", "distant choir",
            "driving ostinato strings", "cinematic climax",
            "wide cinematic stereo image", "quiet verses",
            "full ensemble in final chorus", "no gang vocals", "no group chant",
        ]
    elif mode in ("epic", "epic_rap"):
        head = [
            "powerful female vocal, clear chest-mix belt, emotional and direct, "
            "lead vocal front and center",
            "cinematic epic Mandarin song, film score",
            f"{bpm} bpm", key, "orchestral strings", "grand piano", "choir",
            "deep sub bass", "wide cinematic stereo image", "huge hall reverb",
            "quiet verses", "full orchestra enters in final chorus",
        ]
        if bpm >= 125:
            head += ["epic battle", "taiko drums", "brass fanfare", "driving ostinato"]
        else:
            head += ["slow cinematic build", "soaring climax"]
        if seed.get("rap"):
            head += ["rap verse", "rhythmic spoken flow", "hard consonants"]
    elif mode == "atmospheric":
        head = [
            "sweet clear young female vocal", "breathy intimate low register",
            "restrained delivery", "Mandarin pop", "dark ambient pop",
            f"slow {seed['bpm']} bpm", seed["key"], "spacious mix",
            "sub bass heartbeat", "soft piano",
        ]
    elif mode == "guofeng":
        head = [
            "clear young female vocal with power", "Mandarin guofeng pop",
            "Chinese ancient-style pop", f"{seed['bpm']} bpm", seed["key"],
            "driving drums", "guqin and strings", "cinematic build",
            "crowd-ready chorus",
        ]
    else:
        head = [
            "sweet clear young female vocal", "bright airy tone",
            "tender intimate delivery", "light breathy timbre", "Mandarin pop",
            f"{seed['bpm']} bpm", seed["key"], "warm piano", "shimmering guitars",
            "cinematic chorus",
        ]
    if mode in ("epic", "epic_rap", "epic_ethereal", "battle_anthem"):
        tail = [
            "cinematic production", "wide dynamic range", "deep low end",
            "expansive stereo image", "orchestral depth", "natural hall reverb",
            "clear human vocal", "natural breath", "precise consonants",
            "expressive legato", "controlled vibrato only at phrase ends",
            "no pop production", "no acoustic guitar", "no sweet soft tone",
            "no vocoder", "no robotic tremolo", "no metallic resonance",
            "no synthetic humming", "no harsh digital noise",
            "no onomatopoeia", "no wordless vocalise",
        ]
    else:
        tail = [
            "clear human vocal", "natural breath", "precise consonants",
            "expressive legato", "controlled vibrato only at phrase ends",
            "melodic development", "dynamic phrasing", "believable emotional delivery",
            "polished modern pop", "no vocoder", "no robotic tremolo",
            "no metallic resonance", "no synthetic humming", "no harsh digital noise",
            "no onomatopoeia", "no wordless vocalise",
        ]
    seen: set[str] = set()
    out: list[str] = []
    for item in head + [s.strip() for s in style_extra.split(",")] + tail:
        item = " ".join(item.split())
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return ", ".join(out)


def write_brief(out_dir: Path, seed: dict, data: dict) -> None:
    slug = seed["slug"]
    lyrics = str(data["lyrics"]).strip() + "\n"
    (out_dir / f"{slug}_lyrics.txt").write_text(lyrics, encoding="utf-8")
    tags = build_tags(seed, str(data.get("style_extra", "")))
    (out_dir / f"{slug}_tags.txt").write_text(tags, encoding="utf-8")
    (out_dir / f"{slug}_brief.json").write_text(
        json.dumps({"seed": seed, "brief": data}, indent=2, ensure_ascii=False),
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="",
                    help="comma-separated slug substrings to regenerate")
    ap.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env")
    ap.add_argument("--rebuild-tags", action="store_true")
    ap.add_argument("--seeds-file", type=Path, default=None,
                    help="JSON list of seed cards; overrides the built-in SEEDS")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = (json.loads(args.seeds_file.read_text(encoding="utf-8"))
             if args.seeds_file else SEEDS)
    meta_path = out_dir / "briefs.json"
    meta: dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if args.rebuild_tags:
        for slug, entry in meta.items():
            seed = entry["seed"]
            data = entry["brief"]
            (out_dir / f"{slug}_tags.txt").write_text(
                build_tags(seed, str(data.get("style_extra", ""))), encoding="utf-8")
            print(f"tags rebuilt: {slug}")
        return 0

    key, base_url = _read_api_key(args.env)
    if not key:
        print(f"no DEEPSEEK_API_KEY in {args.env}", file=sys.stderr)
        return 2

    import httpx

    used: list[dict] = [
        {"slug": slug, "title": e["brief"]["title"], "hook": e["brief"]["hook"],
         "first_line": next((ln for ln in e["brief"]["lyrics"].splitlines()
                             if ln.strip() and not ln.startswith("[")), "")}
        for slug, e in meta.items()
    ]
    if args.only:
        needles = [p.strip() for p in args.only.split(",") if p.strip()]
        pending = [s for s in seeds[:args.count]
                   if any(needle in s["slug"] for needle in needles)]
    else:
        pending = [s for s in seeds[:args.count]
                   if args.force or s["slug"] not in meta]
    print(f"writing {len(pending)} brief(s) to {out_dir}")
    with httpx.Client(timeout=180) as client:
        for seed in pending:
            problems: list[str] = []
            data = None
            for attempt in range(1, 6):
                used_for_seed = [u for u in used if u.get("slug") != seed["slug"]]
                data = request_brief(client, base_url, key, seed, used_for_seed, problems)
                problems = validate_brief(data, seed)
                if not problems:
                    break
                print(f"  {seed['slug']} attempt {attempt}: {len(problems)} problem(s)")
                for problem in problems[:4]:
                    print(f"    - {problem}")
            if problems:
                print(f"  SKIP {seed['slug']}: failed local gate after 5 attempts")
                continue
            write_brief(out_dir, seed, data)
            meta[seed["slug"]] = {"seed": seed, "brief": data}
            used = [u for u in used if u.get("slug") != seed["slug"]]
            used.append({"slug": seed["slug"], "title": data["title"],
                         "hook": data["hook"],
                         "first_line": next((ln for ln in data["lyrics"].splitlines()
                                             if ln.strip() and not ln.startswith("[")), "")})
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"  wrote {seed['slug']}: {data['title']} / hook {data['hook']!r}")

    (out_dir / "_titles.json").write_text(
        json.dumps({slug: e["brief"]["title"] for slug, e in meta.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "_langs.json").write_text(
        json.dumps({slug: "zh" for slug in meta}, indent=2), encoding="utf-8")
    (out_dir / "_modes.json").write_text(
        json.dumps({slug: e["seed"]["mode"] for slug, e in meta.items()},
                   indent=2), encoding="utf-8")
    print(f"done: {len(meta)} brief(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
