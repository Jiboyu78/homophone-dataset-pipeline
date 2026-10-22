---
name: zh-rebus-design
description: 从中文词表取答案词，为每个词设计"会意式单图"谜面，并写出可直接送文生图模型的图片描述。用于制作或扩充「看图猜成语／看图猜词语」数据集、挑选可入画的答案词、撰写或检查谜面与生图提示词、校验谜面 JSONL。包含 8 种构图范式的判定规则、元素词典用法、字形参与式的启用条件，以及 SD1.5／SDXL／云模型三种提示词风格的写法。
---

# 会意式单图谜面设计

把「一个中文词」变成「一张能不靠文字就被猜出来的图」。
产物是一份 JSONL：每题包含答案、构图范式、元素表、图片描述、负向词、难度。

这个 skill 只负责 **取词 → 设计谜面 → 写图片描述**。
把图片描述交给文生图模型出图是最后一步，用户自己接（同一份 `prompt_en` 可直接用于
ComfyUI / SD WebUI / 通义万相 / 即梦 / Midjourney）。

## 适用与前置

当任务涉及以下任一情形时加载本 skill：
- 制作或扩充「看图猜成语／看图猜词语」数据集
- 从 JSON 词表里挑适合做会意图解的答案词
- 为某个成语/词语设计谜面、撰写生图提示词
- 校验一批谜面 JSONL 是否合规

前置：本 skill 与同仓库的 `pipeline` 包同源（`pipeline/lexicon.py` 是元素词典，
`pipeline/pinyin.py` 做拼音规范化）。脚本会自动向上找到仓库根目录并 import 它。
状态库 `output/pipeline.db` 里已有 22,000+ 词条的可图化打分与语素切分，优先用它。

本 skill 目录内容：

```text
SKILL.md                     本文件（流程、判定规则、自检清单）
scripts/pick_words.py        第 1 步：取候选答案词 / 查某个字的义项
scripts/validate_rebus.py    第 4 步：机械校验谜面 JSONL
examples/demo_output.jsonl   7 条合格范例（覆盖 7 种构图范式，可直接当模板抄）
examples/bad_output.jsonl    5 条故意写错的记录；跑校验应报 13 个 ERROR，
                             用来确认校验器没坏（一个什么都放行的校验器没有价值）
```

## 四步流程

### 第 1 步：取候选答案词

```text
<python> <skill-directory>/scripts/pick_words.py --db output/pipeline.db --limit 40 --out candidates.json
```

> `<skill-directory>` 是本 skill 被加载后的资源根目录；`<python>` 用 `load_workspace_dependencies`
> 返回的解释器，或直接用仓库里已装好依赖的那个 Python。

输出每条包含设计谜面所需的**全部客观信息**，不要自己猜哪些字能画：

| 字段 | 含义 |
| --- | --- |
| `drawable_score` | 可图化分（0–1），越高越适合做会意谜面 |
| `composition_hint` | 规则推荐的构图范式。可以改，但改了要在 `rationale` 里给理由 |
| `segments` | 语素切分：`char` / `kind` / `en`（英文提示词）/ `drawable` / `sense_forced` |
| `flags` | 多义字风险 / 否定字 / 词典外字 / 可考虑字形参与 |
| `glyph_candidates` | 可走「字形参与式」的字（承载语义但画不出来） |

参数：`--type idiom|word`、`--min-score`、`--order random`、`--offset`（分批取）、
`--words <原始JSON>`（没有 pipeline 状态库时直接从词表取）。
随时可以查某个字的义项：`pick_words.py --lookup 面 心 云`。

**先读 candidates.json，再逐条设计。** 不要跳过这一步凭印象写。

### 第 2 步：设计谜面（核心判断）

对每个答案词，按顺序做四个判断：

**(a) 先看释义，再定元素 —— 这是最容易出错的地方。**
汉字多义，字面直译经常完全跑偏。真实翻车例子：
- `人面兽心` 的「面」如果取"面条"义 → 画面变成 人+面条+兽+心，与题意无关
- `人云亦云` 的「云」是"说"不是"云彩" → 画云就猜不出来
- `胸有成竹` 不是"胸口有根竹子"，但**谜面恰恰要画成胸口有竹子**（会意题的乐趣在此）
- `光阴似箭` 的「光阴」是"时间" → 画光+阴+箭只是字面，需要靠"箭穿过光影"暗示时光飞逝
判断方法：若词表里有 `meaning`（释义）字段，**必须读**；没有释义时，对含多义字的词
保守处理，把不确定写进 `rationale`，不要硬编。

**(b) 选构图范式**（`composition`，八选一）：

| 范式 | 什么时候用 | 画面机制（必须写进提示词） |
| --- | --- | --- |
| `literal` 字面直译 | 复合词逐字具象 | 两个语素合成**一个**物体：火车=火+车、河马=河里的马、冰箱=冰做的箱 |
| `juxtapose` 并列 | 多个独立事物相加 | 同框并列、均匀分布 |
| `spatial` 方位 | 有 上/下/中/里/间/前/后 | 明确分区（上半/下半、里/外） |
| `quantity` 数量 | 有 一/二/三…/半/双/满 | 写出数量与**排列方式**（网格/一列），并指明那个"单独的一个"在哪 |
| `contrast` 反义 | 有 大小/长短/有无/进退… | 左右对照、镜像 |
| `action` 动作 | 有动词（举/守/待/画/添…） | 主体+动作+对象，动作在画面中心且醒目 |
| `inner` 表里 | 有 面/心、表/里、名/实… | **同一个人的影子/剖开处**是另一副样子 |
| `metaphor` 具象隐喻 | 全抽象、画不出具体元素 | 用一个具体物象征抽象概念 |

**(c) 定元素表。** 硬规则（机械校验会拦）：
- 元素必须是答案里**真实出现**的字/词，且词典判为可画（`segments[].drawable`）
- **禁止**把虚词（之/是/而/其）当元素；**禁止**把数词（九/一）、方位词（上/中）
  单列成元素 —— 它们是修饰语，要落到 `count` 或版面描述里
- 每个元素给 `en`：`a/an + 形容词 + 名词` 的具体名词短语
  （`a green coiled snake` ✓；`snake`、`the concept of...` ✗）
- 重复的同类元素用 `count`，不要写两遍
- 词典外字（`flags` 里有"词典外字"）要自己补 `en`

**(d) 判断是否要「字形参与式」**（`technique: "glyph"`）：
只有当词里有**承载语义、却画不出来**的实义字时才用，例如 举**手之劳**的「劳」、
见**仁**见**智**的「智」。判据：该字在 `glyph_candidates` 里。
虚词不算（`马首是瞻` 的「是」不能做字形）。
`glyph` 字段填那**一个**汉字；图片描述里写
`one single large brush-written chinese character <字>`，
并且**不要**在负向词里禁汉字。若你自己接出图工具，
推荐做法是生成时留白、出图后用字体把字叠上去（扩散模型写汉字经常写错）。

**(e) 难度 `difficulty`**：1–5。元素多、含否定字、纯抽象、字长为 5 以上 → 越高。

### 第 3 步：写图片描述

先选风格（`style` 字段）：`tags`（SD1.5 系底模）/ `prose`（SDXL、云模型、中文模型）。

**tags（推荐给本地 SD1.5）**：逗号分隔的短标签，**前 20 个 token 决定主体**，总长 ≤ 40 词。
顺序必须是：`主体+数量+版面` → `风格` → `禁止文字`。例：

```text
9 oxen arranged in a 3 by 3 grid filling the left side, one single hair alone on the far
right side, flat cartoon illustration, bold black outlines, warm cream paper texture
background, no text, no letters, no watermark, single image
```

**prose（SDXL / 云模型）**：可以写完整句子，但要保持"一个画面、一个焦点"：

```text
a smiling young man standing on the left looking completely ordinary, the long dark
shadow behind him is a horned beast with claws and a red heart in its chest, flat
cartoon illustration, bold outlines, warm cream paper background, no text of any kind
```

**所有模式都必须给 `negative_prompt`**，且必须含文字类负向词（否则模型爱乱写字）：
`text, words, letters, chinese characters, calligraphy, caption, watermark, signature,
logo, stamp, seal, number, speech bubble, multiple panels, collage, blurry, low quality`

三条实测结论，直接影响写法：
1. **结构必须显式写出来**。只写"九头牛和一根毛"会得到一大群牛；必须写
   `3 by 3 grid`、`upper half / lower half`、`his shadow is ...`。
2. **SD1.5 不会数数、也不会排网格**。数量式即使写清网格也常失败；
   要精确版面就得在出图环节用 ControlNet（线稿/分割图）或程序化拼贴，
   提示词只能提高概率。设计时对数量式要预期这一风险。
3. **SD1.5 读不懂长句**。同一道题，散文版提示词会画出错误的构图，
   标签版（主体前置）明显更好。风格选错等于白设计。

### 第 4 步：机械校验

```text
<python> <skill-directory>/scripts/validate_rebus.py out.jsonl --report report.md
```

它检查：字段完整、枚举合法、**元素是否真实存在于答案中且可画**、
是否误把虚词/数词/方位词当元素、字形是否合法、提示词是否声明禁止文字、
负向词是否含文字类词、结构性构图是否写出面版、must_have 与元素是否一致、
难度范围、干扰项是否含答案。退出码 0 通过 / 1 有 ERROR / 2 只有 WARNING。
**ERROR 必须改完再交付**；WARNING 逐条判断，不要无脑忽略。

## 输出格式（JSONL，一行一题）

```json
{
  "answer": "九牛一毛",
  "pinyin": "jiǔ niú yī máo",
  "type": "idiom",
  "composition": "quantity",
  "technique": "ideographic",
  "glyph": "",
  "panels": 1,
  "difficulty": 2,
  "elements": [
    {"morpheme": "牛", "zh": "牛", "en": "white oxen", "count": 9,
     "role": "object", "position": "左侧 3×3 网格"},
    {"morpheme": "毛", "zh": "毛", "en": "a single thin hair", "count": 1,
     "role": "object", "position": "右侧独立"}
  ],
  "layout": "左侧 3×3 排布九头牛，右侧单独一根细毛，体量对比是谜眼",
  "prompt_en": "9 oxen arranged in a 3 by 3 grid filling the left side, one single hair alone on the far right side, flat cartoon illustration, bold black outlines, warm cream paper texture background, no text, no letters, no watermark, single image",
  "negative_prompt": "text, words, letters, chinese characters, watermark, signature, multiple panels, collage, blurry, low quality",
  "must_have": ["white oxen", "a single thin hair"],
  "rationale": "数量对比就是谜眼：九与一的悬殊让'极小的部分'这个意思一眼可读",
  "distractors": ["一石二鸟", "九死一生", "凤毛麟角"],
  "style": "tags"
}
```

## 参考基准（用户提供的真实题目，设计时用来校准）

| 答案 | 参考图机制 | 对应范式 | 手法 |
| --- | --- | --- | --- |
| 举手之劳 | 画面中央巨大「劳」字 + 一只举起的手 | `action` | `glyph`（写字） |
| 九牛一毛 | 九头牛排成三行三列 + 一侧一根细线 | `quantity` | `ideographic` |
| 人面兽心 | 男子微笑站立，**影子**是长角利爪、胸口红心的黑色怪物 | `inner` | `ideographic` |
| 马首是瞻 | 前排大兵高举马头形木槌，三名小兵跟随仰望 | `action` | `ideographic` |

设计新题时问自己：**它和这四张的机制像不像？** 如果是新的机制，说明有一处判断可能错了。

## 交付前自检清单

逐条打勾，全过再交付：

- [ ] 每个词的 `meaning`（若有）都读过，多义字没有按字面直译错
- [ ] `composition` 与画面机制一致，`rationale` 说明了"为什么能猜出来"
- [ ] 每个元素都真实出现在答案里，且不是虚词/数词/方位词
- [ ] 元素 `en` 都是具体名词短语，重复元素用 `count` 而不是重复条目
- [ ] 非字形模式：提示词声明了禁止文字，负向词含文字/水印类
- [ ] 字形模式：`glyph` 是单个汉字、在 `glyph_candidates` 里、负向词**未**禁汉字
- [ ] 结构性构图（数量/方位/表里/对比/直译）都显式写出了版面
- [ ] `prompt_style` 与出图后端匹配（本地 SD1.5 用 `tags`）
- [ ] `validate_rebus.py` 退出码为 0（WARNING 已逐条处理）
- [ ] 干扰项 3 个、不含答案、与答案同长度优先

## 已知局限（不要掩盖）

- 本 skill 保证的是**谜面设计质量**，不保证出图质量；出图还受底模、步数、
  ControlNet 影响。数量类构图和"表里/影子"类构图对底模能力要求较高。
- 无释义字段时，多义字只能靠常识判断；写进 `rationale` 便于人工复核。
- 扩散模型仍可能自行写出伪文字。若你的流程里有 OCR/VLM 质检，
  把「画面不得出现计划外文字」作为硬门槛；仅靠提示词无法完全杜绝。
