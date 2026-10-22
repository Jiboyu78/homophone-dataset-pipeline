# 会意猜谜数据集生成 Pipeline

用 AI 批量生产 **「会意式单图猜成语 / 猜词语」** 数据集的流水线：  
把词条拆成可入画的元素 → 组合成具体构图 → 文生图 → 自动质检（含"机器能否猜出"测试）  
→ 打包成带答案、干扰项、难度、质检分的 JSONL 数据集 + 人工复核页。

面向 **离线部署**：只依赖 Python 标准库 + Pillow，生图/质检后端全部可插拔，没有 GPU、没有大模型也能把全流程跑通。

---

## 0. 一键生成（推荐入口）

只需要两个模型的 **API Key + 请求地址**，其余全自动：

```powershell
# 双击 scripts\生成数据集.bat，或命令行：
.\scripts\oneclick.ps1                    # 交互式问你 6 个值，然后全自动跑完
.\scripts\oneclick.ps1 -Limit 200         # 生成 200 条
.\scripts\oneclick.ps1 -Reuse             # 复用上次保存的 key，不再提问
.\scripts\oneclick.ps1 -Stages ingest,select   # 只筛选不出题（不花钱，先看候选质量）
```

它会依次问你要：

| 你要提供            | 说明                               | 例子                                              |
| --------------- | -------------------------------- | ----------------------------------------------- |
| 描述模型 地址/Key/模型名 | 挑词、设计谜面、写图片描述（OpenAI 兼容 chat 接口） | `https://api.deepseek.com/v1` + `deepseek-chat` |
| 出图模型 地址/Key/模型名 | 文生图                              | `https://api.openai.com/v1` + `gpt-image-1`     |

### 怎么运行（Windows 上最容易卡在这里）

**先说最容易踩的一个坑：Windows 上命令是 `python`，不是 `python3`。**  
`python3` 通常指向 `%LOCALAPPDATA%\Microsoft\WindowsApps\python3.exe`，那是个  
**0 字节的微软商店别名存根**；没装商店版 Python 时执行它**什么都不做**——  
既没有输出也没有报错，看起来就像"没有任何反应"。可以用 `python --version` 验证。

`.ps1` **也不是** Windows 的可执行文件类型。在 **cmd** 或资源管理器里直接输入/双击  
`xxx.ps1`，系统是去"打开"它而不是运行它——通常弹出「选择此文件的打开方式」。  
而且本机 PowerShell 执行策略是 `Restricted`，就算你进了 PowerShell 窗口，  
直接 `.\xxx.ps1` 也会被拒（提示"禁止运行脚本"）。

可行方式，任选其一：

```powershell
# 方式 1（推荐：自动挑选可用的解释器——优先 64 位那份，自带 Pillow 与 OCR）
.\scripts\oneclick.ps1 -Limit 10
# 或双击 / 在 cmd 里：
scripts\生成数据集.bat --limit 10

# 方式 2（显式绕开执行策略）
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\oneclick.ps1 -Limit 10

# 方式 3（一次性放开当前用户的执行策略，之后可在 PowerShell 里直接 .\ 运行）
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\oneclick.ps1 -Limit 10

# 方式 4（直接用 Python 跑，注意是 python 不是 python3）
python scripts\oneclick.py --limit 10
```

**解释器选择会影响质检能力**，值得知道：

| 解释器                                       | Pillow（叠字/联系表） | OCR（文字泄漏检测）                 |
| ----------------------------------------- | -------------- | --------------------------- |
| 64 位 Python（如 DSH 自带的 3.12 + 项目内 `.deps`） | ✓              | ✓                           |
| 32 位 Python                               | 可装             | ✗ onnxruntime 没有 32 位 wheel |

用 32 位解释器时脚本会明确提示这一点，质检会自动降级为"技术检查"并在报告里  
注明"未做文字核验"，不会假装检查过。想用完整质检就换 64 位 Python，  
或直接用 `.\scripts\oneclick.ps1`（它优先挑 64 位那份）。

退出码含义（便于写脚本串联）：`0` 成功；`2` 配置缺失；`3` 描述模型连不上；  
`4` 一张图都没生成（通常是出图后端配置不对）。

### 省钱路线：只出图片描述，图拿到别处生成

批量出图走 API 很贵。可以先**只让大模型把描述写完**，把清单交给已经充了会员的  
桌面端生图工具批量出图，再导回流水线继续质检与打包 —— 全程只付出题那点费用。

```bash
# 1) 只出描述，不出图（不会调用出图模型，因此不产生出图费用）
python3 scripts/oneclick.py --limit 500 --no-images

# 2) 把 output/image_tasks.csv 交给你的生图工具
#    按清单里的「文件名」列命名输出文件（文件名里带稳定 id，导回时能对上答案）

# 3) 把生成的图片导回，自动完成 格式统一 → 匹配答案 → 叠字 → 登记 → 质检 → 打包
python3 scripts/oneclick.py --import-images <你放图片的目录>
```

第 1 步会产出（`output/` 下）：

| 文件                      | 用途                                         |
| ----------------------- | ------------------------------------------ |
| `image_tasks.csv`       | 表格：序号、文件名、答案、提示词、负向词、尺寸、需叠加的字、是否降级、必须出现的元素 |
| `image_tasks.jsonl`     | 机读版，字段最全                                   |
| `prompts/*.txt`         | 每条一个纯文本文件，只含提示词（支持"按文件批量读提示词"的工具可直接用）      |
| `README_image_tasks.md` | 给协作者／桌面 agent 的说明                          |

第 3 步会自动处理三件容易出错的事：  
**格式统一成 PNG**（很多工具默认输出 jpg/webp）、  
**字形参与式的汉字由字体确定性叠加**（别指望扩散模型写对汉字，出图时留白即可）、  
**按文件名匹配答案**（匹配不上的会明确列出来，不会静默丢）。

想先不花钱试跑整条闭环，可以拿占位图当"外部生成的图片"：

```bash
python3 scripts/oneclick.py --limit 6 --no-images
python3 scripts/make_placeholder_images.py output/image_tasks.csv /tmp/fake_imgs 6
python3 scripts/oneclick.py --import-images /tmp/fake_imgs
```

### 多路并发：几万条也能跑得快

`--limit` 想写多少写多少（这份词表的上限是 **8,885 条**）。单进程的瓶颈是  
**API 等待**而不是本机算力，所以要多路并发：

```powershell
# Windows（PowerShell 入口）
.\scripts\oneclick.ps1 -Limit 4000 -Workers 4 -NoImages
# 等价写法（两种都认；注意别用 python 去跑 .ps1）
.\scripts\oneclick.ps1 --limit 4000 --workers 4 --no-images

# macOS / Linux
python3 scripts/oneclick.py --limit 4000 --workers 4 --no-images
```

它自动做完了这些事：

- **切片**：每路拿到互不重复的词。按"每隔 N 个取一个"切，而不是连续切块——  
  候选是按分数排序的，连续切块会让 1 号路拿走全部高分词。
- **每路独立输出目录与独立状态库**（`output/shard1..4`）。共用一个状态库是不行的：  
  4 个进程会互相抢锁，而且分片会互相把对方的候选标成 `rejected`，切片直接失效。
- **日志各写各的**（`output/shardN/console.log`），主进程只报心跳，否则 4 路输出糊成一团。
- **跑完自动合并**到 `output/`：任务清单重新连续编号，**状态库也一起并**  
  （items + plans），所以合并后的目录能独立走完"导回图片 → 质检 → 打包"。

不想用 `--workers` 时也可以手工分片：

```bash
python3 -m pipeline.cli --config config.run.shard1.json run --shard 1/4 \
    --stages ingest,select,plan,tasks --limit 1000
# ... 2/4、3/4、4/4 各起一个进程
python3 -m pipeline.cli merge --dirs output/shard1,output/shard2,output/shard3,output/shard4 --out output
```

并发路数建议 4~8。瓶颈通常在服务商的**速率限制**，不在本机；  
路数太多反而会撞限流（报 429/504），这时加大路数会更慢。

### 连不上描述模型？先分层诊断

退出码 `3` 表示描述模型不可用。**不要急着怀疑 key**——先用这个脚本看清卡在哪一层：

```bash
python3 scripts/check_endpoint.py https://你的地址/v1 --key sk-xxx --model 模型名
```

它依次测 DNS → TCP → TLS 握手 → HTTP 请求 →（给了 key 时）真实 chat 调用，  
并给出结论。对照处理：

| 诊断结果                                    | 含义                                                                                                | 处理                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------- |
| TLS 握手就失败                               | 中间有代理/拦截，或地址写错                                                                                    | 换地址                    |
| **`SSL: UNEXPECTED_EOF_WHILE_READING`** | TCP 与 TLS 都正常，但 **HTTP 请求一发出连接就被切断** —— 该域名在你当前网络下被拦截（SNI/内容层），或服务端拒绝非浏览器客户端。**不是 key 的问题，也不是超时** | 换服务商地址                 |
| `HTTP 401/403`                          | 网络完全正常                                                                                            | key 或模型名不对             |
| 60 秒后 `504 Gateway Time-out`            | 网关对长请求超时                                                                                          | 调小每批条数：`--llm-batch 1` |
| 请求发出后被切断/超时，但 `/models` 能通              | 该网关对 chat 长请求会中断                                                                                  | 换服务商                   |

本机实测可直连的参考端点：`https://api.deepseek.com/v1`、  
`https://dashscope.aliyuncs.com/compatible-mode/v1`。

### macOS / Linux

**可以直接用，无需改代码。** 入口是跨平台的 Python 启动器，三个平台共用同一份实现  
（`scripts/oneclick.py`），PowerShell 与 shell 脚本都只是薄壳：

| 平台                  | 运行方式                                                                 |
| ------------------- | -------------------------------------------------------------------- |
| macOS / Linux       | `python3 scripts/oneclick.py --limit 100`                            |
| macOS / Linux（等价）   | `bash scripts/oneclick.sh --limit 100`                               |
| macOS 双击            | Finder 里双击 `scripts/生成数据集.command`（首次先 `chmod +x`）                   |
| Windows（推荐）         | `.\scripts\oneclick.ps1 -Limit 100`，或双击 `scripts\生成数据集.bat`          |
| Windows（直接跑 Python） | `python scripts\oneclick.py --limit 100`　**是 `python`，不是 `python3`** |

> **两个脚本别混着用**：`oneclick.ps1` 是 PowerShell 脚本，必须用  
> `.\scripts\oneclick.ps1` 运行；用 `python` 去执行它会报  
> `SyntaxError: invalid syntax`（Python 在按自己的语法解析 PowerShell 代码）。  
> 反过来也一样：`oneclick.py` 用 `python` 运行。  
> Windows 入口的两种参数写法（`-NoImages` 与 `--no-images`）都支持。

> Windows 上 `python3` 是微软商店的 0 字节别名存根，执行它**没有任何反应**  
> （无输出、无报错）。务必用 `python`，或用上面的 `.ps1` / `.bat` 入口。

macOS 快速开始：

```bash
# 1) 确认 Python（macOS 自带 3.9+；没有就用 brew install python）
python3 --version

# 2) 可选依赖：Pillow（字形叠加/联系表）与 OCR（文字泄漏检测）
python3 -m pip install --user pillow
python3 -m pip install --user rapidocr-onnxruntime    # 约 250MB，纯 CPU 即可

# 3) 跑（会交互式问你要两个模型的 key 与地址）
python3 scripts/oneclick.py --limit 50
```

说明：

- **不需要 GPU、不需要本地大模型**。文字由云端大模型负责，图片由云端文生图负责，  
  两边都走 HTTP，Apple Silicon 与 Intel Mac 都一样。
- **中文字体自动适配**：字形参与式（如「举手之劳」的「劳」）与联系表需要中文字体，  
  代码按平台查找（macOS 用自带的楷体/宋体/苹方），也可以用环境变量指定：  
  `export HV_CJK_FONT=/System/Library/Fonts/Supplemental/Kaiti.ttc`  
  找不到字体时会给出该平台的安装建议，而不是静默出错。
- **OCR 找不到时自动降级**：质检退化为纯技术检查（分辨率/近似重复），  
  并在报告里明确写"未做文字核验"，不会假装检查过。
- **Windows 专属的辅助脚本**（`check_sd_env.py` / `mem_report.py` / `sd_status.py` /  
  `test_sd_speed.py` / `demo_real_sd.py` / `sd_retry_round.py`）只服务于"本机装  
  SD WebUI 出图"那条已经被放弃的路线，在 macOS 上不要用；走云端 API 用不到它们。
- 如果从 Windows 拷贝过来，注意把 `.sh` / `.command` 的可执行位加上：  
  `chmod +x scripts/oneclick.sh scripts/生成数据集.command`

出图接口支持两种形态（`-ImageMode` 选择，或交互时回答）：

- `openai`（默认）：`POST {地址}/images/generations`，兼容 OpenAI、火山方舟 Seedream、  
  硅基流动、各类 one-api/302 网关等"OpenAI 兼容"的出图服务；
- `dashscope`：阿里云百炼「通义万相」的异步接口（提交任务 + 轮询取图）。

**费用是可控的**：词表会全量读取并打分（本地计算，不花钱），但只有  
`目标条数 × 3`（下限 40）条会送去大模型出题；出题通过的才调用出图模型。  
想先看候选词质量、暂时不花钱，就跑 `-Stages ingest,select`。

跑完直接在 `output/` 拿到成品：

```text
data_card.md                    数据卡：字段说明、分布、样例、许可提示
dataset.jsonl                   全量记录（字段最全）
train|val|test.jsonl            按 8:1:1 划分
sft_chat_train.jsonl            多模态对话 SFT（user 带图 + assistant 答答案）
multiple_choice_train.jsonl     选择题（选项已打乱，含 answer_index）
llava_train.jsonl               LLaVA conversations 格式
classification_train.jsonl      图像分类（配 labels.json）
manifest.csv / stats.md / review.html   清单 / 统计 / 人工复核页
images/*.png                    图片
```

`scripts/test_oneclick.py` 是一条**不需要任何 API Key** 的端到端自测：它用本地  
mock 服务仿真"描述模型 + 出图模型"（含 openai 与 dashscope 两种出图形态），  
确认出题、出图、质检、训练格式导出全部可用：

```powershell
python scripts\test_oneclick.py --limit 12 --mode openai
python scripts\test_oneclick.py --limit 8  --mode dashscope
```

### 两个实现上的坑（已修，别再踩）

1. **`.ps1` 必须存成带 BOM 的 UTF-8**。本机 `powershell.exe` 是 5.1，它把无 BOM 的  
   UTF-8 当 GBK 读，中文字符串变成乱码字节，进而破坏引号/括号配对，报出一堆  
   "缺少右 }"的假语法错误。改脚本后记得保持 BOM。
2. **用户配置与运行配置必须分两个文件**：`config.user.json` 存你的 key（生成器绝不覆盖），  
   `config.run.json` 是每次运行生成的完整配置。写成一个文件时，第一次运行就会把  
   key 覆盖成 null，之后 `-Reuse` 再也拿不到。

---

## 1. 它到底做了什么

以「画蛇添足」为例，流水线会自动产出：

| 阶段       | 产物                                                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| select   | 可图化 1.00 / 覆盖率 1.00 / 构图范式 = 动作式                                                                                              |
| plan     | 元素：画（笔刷作画）、蛇、脚；`scene = a simple human figure who paints with a brush a green coiled snake, and then adds an extra bare foot` |
| generate | 一张矢量插画                                                                                                                        |
| qc       | OCR 判定"无文字"、VLM 判定"蛇/脚齐备"、盲猜测试能否猜出「画蛇添足」                                                                                      |
| package  | JSONL 一条记录：答案、拼音、难度、3 个干扰项、元素表、提示词、质检明细                                                                                       |

### 支持的 8 种会意构图范式

| 范式              | 含义                    | 例                      |
| --------------- | --------------------- | ---------------------- |
| `literal` 字面直译  | 复合词逐字具象成一个场景          | 火车=火+车、河马=河里的马、冰箱=冰做的箱 |
| `juxtapose` 并列式 | 多个元素同框并列              | 杯弓蛇影、刀光剑影              |
| `spatial` 方位式   | 元素间位置关系本身表意           | 火上浇油、壶中日月、瓜田李下         |
| `quantity` 数量式  | 数量对比 / 累加（多个元素自动排成网格） | 三长两短、一石二鸟、九牛一毛         |
| `contrast` 反义并置 | 对立元素左右对照              | 有眼无珠                   |
| `action` 动作式    | 主体 + 动作 + 对象          | 画蛇添足、守株待兔、马首是瞻         |
| `inner` 表里式     | 同一主体的"外表"与"内心/影子"并置   | 人面兽心（微笑的人 + 影子是猛兽）     |
| `metaphor` 具象隐喻 | 用具体物替代抽象概念（需 LLM 出题）  | 光阴似箭                   |

### 两种出题手法（`technique` 字段）

| 手法                 | 说明                          | 例                    |
| ------------------ | --------------------------- | -------------------- |
| `ideographic` 会意图解 | 纯图形组合表意，画面中**不得出现任何文字**     | 九牛一毛、人面兽心、马首是瞻       |
| `glyph` 字形参与       | 画面中**故意写入一个汉字**，字形本身是谜面的一部分 | 举手之劳（巨大的「劳」+ 一只举起的手） |

字形参与式默认走**确定性合成**：生成时让模型在指定位置留白，出图后用字体（行楷/楷体）  
把字叠上去——本地 SD 写汉字基本必然写错，这样结果 100% 可控；质检会认可这个字、  
只拦截"计划外"的文字。想让模型自己写就把 `image.glyph_overlay.mode` 改成 `prompt`  
（云模型如通义万相/Seedream 写中文较可靠）。

---

## 2. 目录结构

```
Language-and-Vision/
├─ config.json                  运行配置（改这个）
├─ config.example.json          全量配置示例（含所有可调项）
├─ 成语大全_整理版(1).json        你的数据表（10,820 条成语）
├─ homophone_answer_vocab.json  你的数据表（11,989 条词语，带"类型"字段）
├─ pipeline/
│  ├─ cli.py                    ★ 命令行入口
│  ├─ runner.py                 阶段编排 + 出图-质检闭环 + health 自检
│  ├─ ingest.py                 JSON 表 → 统一 schema（字段自动识别）
│  ├─ select.py                 可图化打分与筛选
│  ├─ lexicon.py                ★ 会意元素词典（1186 条语素，决定"能画什么"）
│  ├─ plan_rule.py              规则出题器（无需任何大模型）
│  ├─ plan_llm.py               LLM 出题器（可选，OpenAI 兼容）
│  ├─ generate.py               批量出图（并发/重试/缓存/反馈回灌）
│  ├─ qc.py                     三路质检 + 定级 + 重试反馈
│  ├─ package.py                JSONL / 划分 / 统计 / 干扰项
│  ├─ review.py                 人工复核页 + 联系表
│  └─ providers/
│     ├─ image_comfyui.py       ComfyUI（推荐）
│     ├─ image_sdwebui.py       SD WebUI / Forge
│     ├─ image_mock.py          Mock：画"构图示意卡"，零依赖跑通全流程
│     ├─ ocr.py                 文字泄漏检测（rapidocr/paddle/easy/tesseract）
│     ├─ vlm_openai.py          视觉质检（可解性测试 + 一致性核验）
│     └─ openai_compat.py       OpenAI 兼容客户端（Ollama/LM Studio/vLLM/云）
├─ workflows/
│  ├─ sd15_txt2img_api.json     SD1.5 ComfyUI 工作流（API 格式）
│  └─ sdxl_txt2img_api.json     SDXL ComfyUI 工作流（API 格式）
├─ scripts/
│  ├─ run.ps1 / run.bat         Windows 启动脚本
│  ├─ selftest.py               ★ 端到端自测（34 条用例，无需 GPU / 大模型）
│  ├─ check_reference.py        ★ 参考题验收：4 张真实题目的机制是否复现
│  ├─ check_ocr.py              验证 OCR 文字泄漏检测是否正常
│  ├─ check_dataset.py          数据集完整性校验（字段/图片/干扰项/划分）
│  ├─ make_reference_demo.py    按 4 张参考题生成对照样例（output_reference/）
│  ├─ demo_full_stack.py        ★ 全栈联调：LLM 出题 + 出图 + OCR/VLM 质检 + 打包
│  └─ mock_llm_server.py        OpenAI 兼容的本地假模型服务（没有真模型时用）
├─ .deps/                       可选依赖（OCR 装在这里，删掉即卸载）
├─ output/                      主产物
├─ output_reference/            4 张参考题的对照样例（含 summary.txt 逐题说明）
└─ output_stackdemo/            全栈联调样例产物（mock 大模型，仅供验证链路）
   ├─ pipeline.db               状态库（断点续跑、幂等、缓存）
   ├─ images/                   生成的图
   ├─ dataset.jsonl / train|val|test.jsonl
   ├─ manifest.csv              可 Excel 打开的总清单
   ├─ stats.md                  统计报告
   ├─ review.html               ★ 人工复核页（浏览器直接打开）
   └─ contact_sheet_*.png       按范式分组的联系表
```


## 4. 验证与自测

在把这条流水线接上真显卡之前，先用三个脚本确认它本身是好的（都不需要 GPU、不需要 API Key）：

```powershell
& $py scripts\selftest.py            # 34 条用例，约 17 秒
& $py scripts\check_reference.py     # 用你给的 4 张真实题目验收机制是否复现
& $py scripts\check_ocr.py           # OCR 文字泄漏检测（真实识别中英文）
& $py scripts\demo_full_stack.py     # 全栈联调：LLM 出题 + 出图 + OCR/VLM 质检 + 打包
& $py scripts\make_reference_demo.py # 按 4 张参考题生成对照图，见 output_reference/
& $py scripts\check_dataset.py output --verify-sha   # 交付前校验数据集自洽
```

### 4.1 参考题对齐（`check_reference.py`）

你提供的 4 张真实题目已固化成验收标准，`check_reference.py` 会逐条断言机制一致：

| 参考题  | 参考图的机制             | pipeline 现在产出                                         | 状态 |
| ---- | ------------------ | ----------------------------------------------------- | -- |
| 举手之劳 | 巨大「劳」字 + 一只举起的手    | `glyph` 手法，`glyph=劳`，预留空白后叠字                          | ✅  |
| 九牛一毛 | 九牛 3×3 网格 + 一侧一根细毛 | `quantity`，牛×9 + 毛×1，提示词写"3 by 3 grid"                | ✅  |
| 人面兽心 | 微笑的人 + 影子是长角猛兽带红心  | `inner` 表里式，外表=人+人脸，内心=兽的心                            | ✅  |
| 马首是瞻 | 高举马头，众人仰望          | `action`，`马首` 合并为"the head of a horse"，`瞻`=gaze up at | ✅  |

这 4 条断言覆盖了参考图暴露出的 4 类真实缺陷（多义字解错、缺表里式、缺字形参与、  
动物+身体部位不合并），改动后已全部修复并纳入回归测试。

跑 `make_reference_demo.py` 可以生成一组对照图（`output_reference/`），  
其中"举手之劳"那张会真的把「劳」字按参考题的样式叠进画面，并给出 `summary.txt`  
逐题列出构图、元素与提示词，方便和你手上的原题并排比对。

`selftest.py` 覆盖的内容：

| 分组        | 断言的东西                                                                                  |
| --------- | -------------------------------------------------------------------------------------- |
| Ingest    | 字段自动识别（中英文字段名）、跨表去重、**重复运行幂等**、ID 稳定性                                                  |
| Lexicon   | 语素切分、多义字义项（香醋的"香"取"芳香的"）、虚词排除、词典指纹                                                     |
| Select    | 同义并列复合词拒绝（米饭=米+饭）、属性对判为数量式、词典变更自动重算                                                    |
| Plan      | 每个接受条目都有方案；提示词不变量（不出现 `identical identical`、冠词不重复、必带"禁止文字"）；must_have 非空、元素不重复、难度在 1–5 |
| Plan(LLM) | 对着 mock 服务跑通批量出题、JSON 解析、`planner=llm` 落库；服务不可用时**自动回退规则版**                            |
| QC        | 空白图判 reject；无 VLM 时技术质检通过即接受；假 VLM 报文字 → **retry 且反馈非空** → 次轮转 accept；到顶仍带文字 → reject  |
| Generate  | 反馈回灌进提示词、prompt_hash 变化、**reject 的条目不再入队浪费算力**                                         |
| ComfyUI   | 工作流注入（正向/负向/尺寸/种子/步数/CFG）、界面格式工作流被明确拒绝                                                 |
| OCR       | 纯几何图形（圆被读成 "O"）**不误报**；中文、英文水印都能检出；弱证据权重计算                                             |
| Package   | 必需字段齐全、图片存在、干扰项恰好 3 个且不含答案、划分已分配                                                       |

`demo_full_stack.py` 会打印一份"链路自检"报告，例如本机实测：

```
  出题器来源分布: {'llm': 4000}
  质检通道分布: {'technical+vlm': 47}
  质检结论分布: {'retry': 25, 'needs_human': 2, 'accept': 19, 'reject': 1}
  可解性被测过的图: 47    文字核验过的图: 47（其中无文字 31）
  触发过重出图的条目: 14  最终数据集记录数: 19
```

这证明了 **LLM 出题、VLM 核验、可解性测试、反馈重出图闭环、OCR、打包** 全部真实执行过。  
其中"mock 大模型"只是接线测试替身，它的答案与猜测没有实际意义。

## 5. 数据集字段

`output/dataset.jsonl` 每行一条：

| 字段                                            | 说明                                                                                              |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `id`                                          | 稳定哈希 ID（跨轮次/跨机器一致，可用于对齐标注）                                                                      |
| `answer`                                      | 谜底（成语或词语）                                                                                       |
| `pinyin`                                      | 带声调符号的拼音（原始数据是 `hua4 she2...`，已自动转换）                                                            |
| `type`                                        | `idiom` 成语 / `word` 词语                                                                          |
| `category`                                    | 原始数据里的"类型"（食物/动物/生活用品…）                                                                         |
| `composition` / `composition_zh`              | 构图范式（见第 1 节）                                                                                    |
| `technique` / `technique_zh`                  | 出题手法：`ideographic` 会意图解 / `glyph` 字形参与 / `homophone` 谐音                                         |
| `glyph`                                       | 字形参与式要写进画面的那个汉字（如 举手之劳 → `劳`），其余为空                                                              |
| `panels`                                      | 画面数，固定 1（单幅）                                                                                    |
| `difficulty`                                  | 1–5，由元素数、多义字、否定字、词长推算                                                                           |
| `image`                                       | 相对 `output/` 的图片路径                                                                              |
| `image_sha1`                                  | 图片内容哈希                                                                                          |
| `elements`                                    | 元素表 `[{morpheme, zh, en, count, role, position}]`                                               |
| `must_have`                                   | 质检/标注时用来核对的关键元素英文短语                                                                             |
| `distractors`                                 | 3 个干扰答案（优先同长度、有共同字、同范式）                                                                         |
| `qc`                                          | `verdict / score / text_free / element_coverage / solvable / guessed_answer / checker / issues` |
| `prompt_en` / `prompt_zh` / `negative_prompt` | 可直接复现该图                                                                                         |
| `layout` / `rationale`                        | 中文构图说明与"为什么能被猜出来"                                                                               |
| `split`                                       | `train` / `val` / `test`                                                                        |
| `flags`                                       | 风险标记：`negation:不`、`polysemy_risk`、`literal_compound` 等                                          |

`solvable` 字段值得单独说明：它是**不告诉模型答案、让视觉模型盲猜**的结果，  
是衡量"这张图到底能不能猜出来"的最直接指标，也是这个数据集区别于普通图文对的核心价值。

---

## 6. 质检与重试闭环

```
        ┌──────────────────────────────────────────────┐
        ↓                                              │
   generate ──图──→ qc ──accept──→ 打包                │
                     │                                 │
                     ├──retry──→ feedback_for_retry ────┘
                     ├──reject（空白/损坏/空白纯色）：不再浪费算力重试
                     └──needs_human：转人工复核页
```

三路质检：

1. **技术质检**（永远可用，Pillow）：分辨率、长宽比、灰度标准差（空白检测）、边缘密度、dHash 近似重复。  
   默认只查"跨词条"重复，因为同一词条的不同尝试本来就该相似；`qc.technical.dedup_scope` 可改。
2. **文字泄漏检测**（已在本机启用，rapidocr）：扩散模型最爱在画面上"写出"伪汉字和水印，  
   这是会意谜面的头号失败模式。检出后**触发重出图**并把"remove every trace of text"  
   回灌进下一轮提示词（而不是直接丢弃——这是可以救回来的）。  
   Mock 后端画的是给人看的"构图示意卡"，其标签通过 `qc.ocr.skip_providers` 豁免。
3. **语义质检**（可选 VLM）：
   - **可解性测试**：只给图、不给答案，让模型猜 → `solvable`
   - **一致性核验**：给答案 + `must_have`，查元素齐备性、文字、多格/水印/裁切

未启用 VLM 时，质检分会被压在 0.85 以内并在 `issues` 里标注"语义未验证"，  
便于你事后按分数筛选。`qc.require_solvable = true` 可要求"机器必须猜得出"才入库。

---

## 7. 常用命令

```powershell
& $py -m pipeline.cli health                      # 自检：数据/生图/LLM/VLM/OCR/词典
& $py -m pipeline.cli run --stages select,plan    # 只出题
& $py -m pipeline.cli run --mock --limit 30       # Mock 后端跑通闭环
& $py -m pipeline.cli run --provider comfyui      # 临时切换生图后端
& $py -m pipeline.cli inspect 画蛇添足             # 分词/打分/构图方案（调词典必备）
& $py -m pipeline.cli stats                       # 状态库统计
& $py -m pipeline.cli prompts --limit 200         # 导出提示词 CSV（可丢给任何出图工具）
& $py -m pipeline.cli review                      # 刷新复核页与联系表
& $py -m pipeline.cli reset --stage generate      # 重置某阶段（ingest/select/plan/generate/qc/all）
& $py -m pipeline.cli apply-decisions --file output\decisions.json   # 落库人工复核结果
```

`--limit N` &#x662F;**"首次出图"的条目数**，不是总张数：已判 retry 的条目重出图不占额度，  
所以 `--limit 500 --rounds 3` = 最多 500 个新词条，每个最多尝试 3 次。  
`--limit-words N` 则限制 `ingest/select/plan` 处理多少词条（默认全部 22,023 条）。

> 也就是说：**「跑 40 条」= `run --limit 40`**，词表照样全量读取与打分（分钟级、零成本），  
> 只是只给 40 条最合适的词出图；这 40 条里质检不过的会带着修正建议自动重出，不额外占额度。

---

## 8. 部署 ComfyUI（Tesla P100 特别注意）

本机是 **Tesla P100 16GB**。P100 的 FP16 吞吐极低（1/64），直接跑默认参数会非常慢，  
必须注意：

```bat
:: SD1.5，推荐加 --fp32-unet 或 --no-half
python main.py --listen 127.0.0.1 --port 8188 --fp32-unet --disable-xformers

:: SDXL 在 P100 上勉强可跑（1024² 约 30–60s/张），建议
python main.py --listen 127.0.0.1 --port 8188 --no-half --medvram
```

要点：

- **显存**：SD1.5 用 768×768 足够（谜面是"象征静物"，不需要高分辨率）；SDXL 用 1024×1024。
- **速度**：单卡串行最快，保持 `image.concurrency = 1`；P100 不要开 xformers/fp16。
- **工作流必须导出成 API 格式**：ComfyUI 菜单 `Settings → Enable Dev mode Options`，  
  然后 `Save (API Format)`。用界面格式（含 `nodes`/`links`）会被 pipeline 明确拒绝并提示。
- 想换工作流：把自己的 API 格式 JSON 丢到 `workflows/`，改 `image.comfyui.workflow` 即可。  
  pipeline 按节点类型自动注入提示词/尺寸/种子/步数，无需改代码；需要精细控制时用  
  `image.comfyui.overrides = {"3": {"cfg": 7.0}}` 按节点 id 覆盖。

### 用 SD WebUI 代替

AUTOMATIC1111 / Forge 更简单，开 `--api`，然后把 `image.provider` 改成 `sdwebui` 即可：

```bat
webui-user.bat 里加： set COMMANDLINE_ARGS=--api --no-half --medvram
```

---

## 9. 提升质量的三条路（按性价比排序）

### ⓪ 画风与画幅（已按参考题对齐）

默认 `plan.style_preset = "rebus_cartoon"`，对齐你给的参考图：**横向 1152×768 单幅、  
粗黑描边卡通、米黄做旧纸纹背景、形象夸张**。另一个预设 `clean_vector` 是原来的  
"简约扁平矢量插画、768×768 方形、留白充足"。

```json
"plan": { "style_preset": "rebus_cartoon" }   // 或 "clean_vector"
```

预设只提供默认值，你在 `config.json` 里显式写的 `plan.style_en` / `plan.width` /  
`plan.height` 仍然优先（想完全自定义就覆盖这三项）。

### ① OCR 文字泄漏检测 —— ✅ 本机已装好

OCR 引擎（rapidocr-onnxruntime，251 MB）已经装在项目内的 `.deps/`，不污染系统 Python：

```powershell
& $py scripts\check_ocr.py       # 验证：纯图形不误报、中文/英文水印都能检出
& $py -m pipeline.cli health     # 应显示 "OCR: OK OCR 引擎：rapidocr"
```

它会把"图里写了伪汉字/水印"的废图自动拦下并触发重出图。  
换机器时只需重跑一次安装（或整目录拷 `.deps`）：

```powershell
& $py -m pip install --target .deps rapidocr-onnxruntime
```

> 实现细节：单个孤立拉丁字符（圆的轮廓常被读成 "O"）只算 0.34 的弱证据，  
> 需要凑够 0.5 以上才判定为"有文字"，所以几何图形不会被误杀。

### ② 打开 VLM 语义质检（下一步最值得做的）

本地跑一个视觉模型（Ollama）：`ollama pull qwen2.5vl:7b`，然后在 `config.json` 改：

```json
"qc": { "vlm": { "provider": "openai", "base_url": "http://127.0.0.1:11434/v1",
                 "model": "qwen2.5vl:7b", "check_solvability": true },
        "require_solvable": false }
```

于是每条数据都会带上"机器能不能猜出来"的结论——这是筛掉废题最有效的手段。  
把 `require_solvable` 改成 `true`，机器盲猜不出的图会直接不入库（更严，产量更低）。

### ③ 用 LLM 出题替代规则出题

规则出题器免费的代价是**多义字消歧**做不到（比如「人云亦云」的"云"不是云彩）。  
本地起一个 LLM（`ollama pull qwen2.5:14b-instruct`），然后：

```json
"plan": { "provider": "llm",
          "llm": { "base_url": "http://127.0.0.1:11434/v1",
                   "model": "qwen2.5:14b-instruct", "batch": 8 } }
```

LLM 还能顺手判定"这个词根本不适合会意化"并自动剔除。调用失败会自动回退规则版，不会中断。

> 想接云端（DeepSeek / 通义 / OpenAI 等）：把 `base_url`、`model` 指向对方，  
> Key 用环境变量 `HV_LLM_API_KEY` / `HV_VLM_API_KEY`，不要把 Key 写进 config.json。

### ④ 扩充词典（长期收益最大）

`pipeline/lexicon.py` 是整个系统的知识底座，格式极简：

```python
"animal": ["蛇|a green coiled snake", "马|horse"],
"object": ["鼎|a bronze tripod cauldron"],
"dimension": ["长|long", "短|short"],
```

加一行就多一批可出题的词。改了词典后，下次 `run` 会自动检测到词典指纹变化并重算 select/plan。

看哪些字还不在词典里（按影响面排序）：

```powershell
& $py -c "import sys,collections;sys.path.insert(0,'.');from pipeline.config import load_config;from pipeline.state import State;from pipeline.lexicon import LEX;cfg=load_config('config.json');st=State(cfg.db_path);c=collections.Counter(s['morpheme'] for i in st.load_items() for s in i.segments if s['kind']=='unknown');print(c.most_common(40))"
```

---

## 10. 复现性与工程保证

- **稳定 ID**：`id = sha1(类型+词条)`，不依赖 Python 的 `hash()`（进程间随机化会导致重复条目）。
- **幂等**：任一步骤可重复执行；`ingest` 内置跨表去重（同词优先保留成语条目）。
- **断点续跑**：SQLite 状态库记录每个条目/图片/质检的状态，中断后重跑只处理未完成的。
- **提示词缓存**：`prompt_hash = sha1(正向词, 负向词, 反馈, 尺寸)`，命中即复用已有图，重跑不烧算力。
- **词典版本联动**：词典指纹变化自动触发 select 重算与 plan 重生成，杜绝"旧义项残留"。
- **失败不阻塞**：单条出图失败只记日志并跳过，不影响整批；LLM/VLM 不可用自动降级。
- **增量累加**：`--limit` 是"本次新增多少条"的预算，已完成的条目不会重做，  
  所以可以反复跑 `run --limit 200` 逐步把数据集做大。
- **取样均衡**：`image.selection_strategy` 默认 `balanced`，按构图范式轮询取样；  
  否则会因为"字面直译式"分数最高而让数据集全是简单复合词。  
  改成 `score` 则纯按可图化分排序。
- **重试不吃预算**：`--limit N` 只限制"首次出图"的词条数，retry 重出不占额度，  
  所以闭环能真正转起来（早期版本第 1 轮就把额度花光，导致 14 条 retry 却 0 条重出图）。
- **提示词硬约束兜底**：无论方案来自规则、LLM 还是人工编辑，生图前都会检查并补上  
  "禁止任何文字"那句；漏了就补，不给文字泄漏留后门。
- **按可修复性定级**：图损坏/空白 → reject（重试无意义）；画面有文字、撞车、元素缺失、  
  机器猜不出 → retry（换种子+反馈就能救）；重试到顶仍带文字 → 淘汰而不是转人工。

---

## 11. 常见问题

**Q：`health` 报 "无法连接 ComfyUI"**  
A：ComfyUI 没启动，或端口不是 8188。先跑 `--mock` 验证流水线本身没问题。

**Q：报 "这看起来是 ComfyUI 的『界面格式』工作流"**  
A：用 `Save (API Format)` 重新导出。判断依据是 JSON 里同时含 `nodes` 与 `links`。

**Q：出图很慢**  
A：P100 别用 fp16（加 `--fp32-unet`/`--no-half`）；把分辨率降到 768；保持 `concurrency=1`；  
先用 `--limit 20` 试跑测速，再决定批量大小。

**Q：大量图被判"有文字"**  
A：这是扩散模型的通病，正常。负向词里已经写了 `text, letters, chinese characters, hanzi, kanji`，  
若仍频繁出现，可在 `config.json` 的 `plan.style_en` 里追加 `poster, sign, book cover, ui`，  
或换一个对文字更不敏感的模型（SD1.5 动漫/扁平插画类底模通常比写实底模更少乱写字）。

**Q：想让答案更难/更简单**  
A：调 `select.min_score`（越高越挑）、`select.max_items`（截断数量）、  
或在打包后按 `difficulty` 字段自行分层取样。

**Q：能把 `output/` 删掉重来吗？**  
A：可以，但要知道 **状态库就在 `output/pipeline.db`**：删掉 `output/` 等于清空所有进度  
（词条、打分、方案、已出图、质检记录全没），下次要重跑 ingest→select→plan（本机约 90 秒）。  
只想清图片就删 `output/images/` 或跑 `reset --stage generate`，别整个删。  
原始词表和 `.deps` 不受影响。

**Q：一批里为什么"字形参与式"占三成？**  
A：那是 `image.max_glyph_share`（默认 0.30）在起作用，取自参考题的经验比例（4 张里约 1 张）。  
全池子里有 565 条字形参与式，占比只有 7%，按纯分数排序会被整批挤掉，所以改用  
"按配额从全池挑选"。想调比例就改这个值（设为 1 则不限）。

**Q：词语表里有一堆地名/明星/品牌，被拒绝了对不对？**  
A：对。`select.CATEGORY_WEIGHTS` 给专名类打了 0.1–0.2 的系数并硬拒——  
地名、明星、品牌没法用"元素组合"表意，只能做谐音题，不在本 pipeline 的会意范畴内。  
想放开就改这个表。

---

## 12. 许可与使用提示

- 生成图片的版权取决于你使用的底模许可（SD1.5/SDXL 系列各有条款），商用前请自行核对。
- 原始词表来自你自己的数据文件，注意其来源与授权。
- `output/review.html` 是纯静态页面，复核决策存在浏览器 localStorage，  
  请定期"导出 decisions.json"再用 `apply-decisions` 落库，避免换浏览器丢失。

---

## 附：谜面设计 skill（给 AI 用）

`skills/zh-rebus-design/` 是一个独立 skill，把「取词 → 设计谜面 → 写出图片描述」  
交给 AI 来做，出图那一步你自己接（同一份 `prompt_en` 可直接用于  
ComfyUI / SD WebUI / 通义万相 / 即梦 / Midjourney）。

它存在的理由：**规则出题器不会读语义**。实测它会把人面兽心的「面」当成"面条"、  
把马首是瞻的「是」当成画面元素、也读不出「人云亦云」的云是"说"。这些判断  
需要模型来读释义+常识，而不是查字表。

```powershell
# 第 1 步：取候选词（含可图化分、语素切分、多义字风险、字形候选）
& $py skills\zh-rebus-design\scripts\pick_words.py --db output\pipeline.db --limit 40 --out candidates.json
& $py skills\zh-rebus-design\scripts\pick_words.py --lookup 面 云 是     # 随时查义项
# 第 2、3 步：AI 读 SKILL.md，逐条设计谜面并写出图片描述（JSONL）
# 第 4 步：机械校验
& $py skills\zh-rebus-design\scripts\validate_rebus.py out.jsonl --report report.md
```

`SKILL.md` 里固化了这一整轮实测出来的东西：8 种构图范式的判定规则、  
「先看释义再定元素」的多义字陷阱、虚词/数词不得当元素的硬规则、  
字形参与式的启用条件与"交给字体叠加而不是让模型写"的结论、  
以及 SD1.5 读不懂长句/不会数数/排不出网格这三条出图边界。  
`examples/` 里 7 条合格范例可直接当模板，5 条错误范例用于自检校验器。

> 想让所有会话都能用 `skill` 工具直接加载它：把 `skills/zh-rebus-design/`  
> 整个目录复制到 DSH 的 skills 根目录（本会话沙箱不允许写该路径，需要你复制  
> 或授权）。放在工作区里则对本仓库的任何 agent 都可见。
