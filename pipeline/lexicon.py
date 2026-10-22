"""会意元素词典：汉字/语素 -> 可入画的视觉元素 + 英文提示词。

这是整条 pipeline 的知识底座，决定了：
  1) select 阶段如何判断一个词条"能不能用元素组合画出来"（可图化打分）；
  2) plan  阶段为每个语素选用什么英文提示词（直接影响出图可控性）。

条目格式： "字|主英文提示词|备用英文提示词"
kind 决定该语素的"可图化权重"与在构图中的默认角色（见 KIND_PROPS）。
"""
from __future__ import annotations

from typing import Any

# kind -> (可图化权重, 角色)
#   draw     = 直接可画成具体视觉元素
#   modifier = 可画但通常作修饰（颜色/数量/方位），需要依附主体
#   weak     = 只能间接表现
#   no       = 抽象虚词，不入画，也不阻塞构图
KIND_PROPS: dict[str, tuple[float, str, str]] = {
    "animal":    (1.00, "object", "draw"),
    "plant":     (1.00, "object", "draw"),
    "food":      (1.00, "object", "draw"),
    "object":    (1.00, "object", "draw"),
    "tool":      (1.00, "object", "draw"),
    "weapon":    (1.00, "object", "draw"),
    "cloth":     (1.00, "object", "draw"),
    "vehicle":   (1.00, "object", "draw"),
    "building":  (0.95, "background", "draw"),
    "place":     (0.90, "background", "draw"),
    "nature":    (1.00, "background", "draw"),
    "sky":       (1.00, "background", "draw"),
    "weather":   (0.95, "background", "draw"),
    "material":  (0.90, "object", "draw"),
    "body":      (0.95, "object", "draw"),
    "organ":     (0.95, "object", "draw"),
    "person":    (0.90, "subject", "draw"),
    "action":    (0.85, "action", "draw"),
    "motion":    (0.80, "action", "draw"),
    "emotion":   (0.70, "action", "draw"),
    # 度量/属性类形容词：本身可画（长物与短物、大物与小物），是"三长两短""大同小异"的关键
    "dimension": (0.72, "modifier", "modifier"),
    "property":  (0.64, "modifier", "modifier"),
    "color":     (0.70, "modifier", "modifier"),
    "number":    (0.70, "modifier", "modifier"),
    "direction": (0.72, "modifier", "modifier"),
    "time":      (0.60, "modifier", "modifier"),
    "measure":   (0.60, "modifier", "modifier"),
    "sound":     (0.55, "modifier", "modifier"),
    "state":     (0.50, "modifier", "weak"),
    "abstract":  (0.12, "modifier", "no"),
    "function":  (0.08, "modifier", "no"),
}

# 虚词/功能字：不算元素，但也不扣分（中文成语大量依赖它们成句）
# 归入这里的字会从"可画语素覆盖率"的分母中剔除，避免"人不为己天诛地灭"这类句子被误判为不可画。
FUNCTION_CHARS = set(
    "不之而为其所以于与也者矣焉哉乎乃则且若夫盖唯惟兮耶邪"
    "是非有在得的地了着过们个把被将给对从向于至以及并且但是因此所以"
    "然必就别论能会应该没常致当才已相互更最甚极太亦又复各每凡诸"
    "因故但并及至知可好成生事物力形字词味色难失求通经定变结息用全"
    "受比依任应顾兴传私顺处代患置逢妙贤敢英久济积忍刻始近养累微易"
    "识引屈贫调错寻带糊威源害载章表杂俗益赏令终点心怀恩"
)

LEXICON_SRC: dict[str, list[str]] = {
    # ------------------------------------------------------------------ 动物
    "animal": [
        "马|horse", "牛|ox", "羊|sheep", "狗|dog", "犬|dog", "猫|cat",
        "猪|pig", "鸡|hen", "鸭|duck", "鹅|goose",
        "鱼|fish", "虎|tiger", "狮|lion",
        "狼|wolf", "狐|fox", "兔|rabbit", "鼠|mouse",
        "蛇|a green coiled snake", "龙|a golden chinese dragon", "凤|a phoenix",
        "鸟|bird", "雀|sparrow", "鹰|eagle", "鹤|a red-crowned crane",
        "燕|swallow", "鸦|crow", "鹊|magpie", "雉|pheasant",
        "猴|monkey", "猿|ape", "熊|bear", "豹|leopard",
        "象|elephant", "鹿|deer", "驴|donkey", "骡|mule",
        "骆|a bactrian camel", "驼|a camel", "鲸|a whale", "鲨|a shark",
        "虾|a pink shrimp", "蟹|a crab", "龟|a turtle", "鳖|a softshell turtle",
        "蛙|a green frog", "蟾|a toad", "蝶|a butterfly", "蛾|a moth",
        "蜂|a honeybee", "蚁|a line of ants", "蝉|a cicada", "蜻|a dragonfly",
        "蚊|a mosquito", "蝇|a fly", "蛛|a spider on a web", "蚕|a silkworm",
        "虫|a small green bug", "兽|a wild beast", "禽|a fowl", "畜|a farm animal",
        "雏|a baby chick", "羔|a lamb", "驹|a foal", "犊|a calf",
        "蝮|a pit viper", "蟒|a python", "蛟|a water dragon", "螳|a praying mantis",
        "鹏|a giant mythical bird", "鸾|a mythical bird", "鸳|a mandarin duck",
        "鸯|a mandarin duck", "鹦|a parrot", "鹉|a parrot", "鸽|a pigeon",
        "鸥|a seagull", "鹭|a heron", "雕|an eagle", "隼|a falcon",
        "鲸|a whale", "豚|a dolphin", "貂|a marten", "獐|a river deer",
        "狈|a wolf-like beast", "貉|a raccoon dog", "獭|an otter", "狸|a raccoon dog",
        "蝗|a locust", "蚜|an aphid", "蛆|a maggot", "虱|a louse",
    ],
    # ------------------------------------------------------------------ 植物
    "plant": [
        "树|a leafy tree", "木|a wooden log", "林|a dense forest", "森|a dark forest",
        "草|a clump of green grass", "花|a blooming flower", "叶|a green leaf",
        "根|exposed tree roots", "枝|a bare branch", "茎|a plant stem", "芽|a sprouting shoot",
        "果|a ripe fruit", "桃|a pink peach", "李|a plum", "梨|a pear",
        "杏|an apricot", "梅|a plum blossom branch", "兰|an orchid", "竹|green bamboo stalks",
        "菊|a chrysanthemum", "松|a pine tree", "柏|a cypress", "柳|a weeping willow",
        "杨|a poplar tree", "槐|a pagoda tree", "桑|a mulberry tree", "梧|a parasol tree",
        "桐|a paulownia tree", "枫|a red maple", "桂|an osmanthus branch", "莲|a lotus flower",
        "荷|a lotus leaf and flower", "萍|duckweed on water", "藻|green algae",
        "苔|moss on stone", "藤|a climbing vine", "蔓|trailing vines", "荆|thorny branches",
        "棘|thorny thistles", "蓬|a tumbleweed", "蒿|wild artemisia", "芦|reeds",
        "苇|reeds by water", "蒲|a cattail", "瓜|a melon", "豆|beans",
        "稻|rice plants", "麦|golden wheat", "禾|a grain stalk", "粟|millet grains",
        "黍|broomcorn millet", "粮|sacks of grain", "菜|a leafy vegetable",
        "麻|hemp fibers", "棉|a cotton boll", "茶|tea leaves", "烟|a wisp of smoke",
        "莓|strawberries", "蔗|sugar cane", "薯|a sweet potato", "萝卜|a white radish",
        "葱|a bunch of scallions", "蒜|garlic bulbs", "姜|ginger root", "椒|red chili peppers",
        "枣|red dates", "栗|chestnuts", "榛|hazelnuts", "核|a hard nut",
    ],
    # ------------------------------------------------------------------ 食物
    "food": [
        "饭|a bowl of steamed rice", "米|a heap of raw rice", "面|a bowl of noodles",
        "粥|a bowl of congee", "汤|a bowl of soup", "菜|a plate of stir-fried greens",
        "肉|a chunk of raw meat", "骨|a bare bone", "蛋|a cracked egg", "奶|a glass of milk",
        "酒|a jug of wine", "饼|a flat pancake", "糕|a piece of cake", "糖|a lump of sugar",
        "盐|a small pile of salt", "醋|a vinegar bottle", "油|a bottle of cooking oil",
        "酱|a jar of sauce", "蜜|a honey jar", "丸|a round meatball", "包|a steamed bun",
        "饺|a dumpling", "菜|vegetables", "酱|sauce", "鲜|fresh food",
        "药|a paper packet of medicine", "丸|a pill", "酒|a rice-wine jar",
    ],
    # ------------------------------------------------------------------ 器物
    "object": [
        "衣|a folded robe", "户|a wooden household door", "堂|a bright ancestral hall",
        "舍|a small thatched hut", "宅|a walled house", "宇|a house roof",
        "墓|a grave mound", "棺|a wooden coffin", "表|a pocket watch",
        "章|a carved stone seal", "宝|a glowing treasure pearl", "财|a pile of gold coins",
        "源|a spring bubbling out of the ground", "尾|an animal tail",
        "株|a cut tree stump", "桩|a wooden stake", "篱|a bamboo fence",
        "仁|a single kernel or nut", "机|a small machine with gears",
        "芯|a plant core", "籽|seeds",
        "本|exposed plant roots", "席|a woven mat", "舟|a small boat",
        "杯|a ceramic cup", "盘|a shallow plate", "碗|a porcelain bowl", "壶|a teapot",
        "瓶|a glass bottle", "罐|a clay jar", "缸|a large water vat", "盆|a washbasin",
        "锅|a wok", "勺|a ladle", "筷|a pair of chopsticks", "叉|a fork",
        "匙|a spoon", "桌|a wooden table", "椅|a wooden chair", "床|a wooden bed",
        "柜|a wooden cabinet", "箱|a wooden chest", "笼|a bamboo cage", "筐|a woven basket",
        "篮|a woven basket", "袋|a cloth bag", "囊|a small pouch", "包|a cloth bundle",
        "伞|an open umbrella", "扇|a folding fan", "灯|an oil lamp", "烛|a burning candle",
        "香|an incense stick", "镜|a bronze mirror", "梳|a wooden comb", "针|a needle",
        "线|a spool of thread", "绳|a coiled rope", "索|a thick rope", "网|a fishing net",
        "钩|a metal hook", "锁|a padlock", "钥|a key", "门|a wooden door",
        "窗|a lattice window", "墙|a brick wall", "瓦|roof tiles", "砖|a stack of bricks",
        "石|a grey boulder", "玉|a translucent jade piece", "珠|a round pearl",
        "金|gold ingots", "银|silver ingots", "铜|a bronze vessel", "铁|a rusted iron bar",
        "钱|a pile of copper coins", "币|coins", "旗|a banner flag", "鼓|a large drum",
        "钟|a bronze bell", "锣|a gong", "琴|a guqin zither", "笛|a bamboo flute",
        "箫|a bamboo flute", "瑟|a stringed instrument", "棋|a go board with stones",
        "笔|a writing brush", "墨|an ink stick", "砚|an inkstone", "纸|a sheet of paper",
        "书|a stack of old books", "卷|a rolled scroll", "册|a bound book", "画|a hanging scroll painting",
        "印|a carved stone seal", "砚|an inkstone", "简|bamboo writing slips",
        "车|a wooden cart", "船|a wooden boat", "舟|a small boat", "轿|a sedan chair",
        "轮|a wooden wheel", "帆|a white sail", "桨|a wooden oar", "锚|an iron anchor",
        "桥|an arched stone bridge", "塔|a tall pagoda", "寺|a temple", "庙|a shrine",
        "宫|a palace hall", "殿|a palace hall", "楼|a two-storey pavilion", "阁|a pavilion",
        "亭|a small pavilion", "台|a stone terrace", "城|a walled city gate", "村|a small village",
        "田|a grid of rice paddies", "井|a stone water well", "塘|a pond", "渠|an irrigation ditch",
        "路|a winding path", "街|a narrow street", "巷|a narrow alley", "院|a courtyard",
        "阶|stone steps", "梯|a wooden ladder", "仓|a granary", "窖|a storage pit",
        "坟|a grave mound", "碑|a stone stele", "鼎|a bronze tripod cauldron", "炉|a small stove",
        "灶|a brick stove", "釜|an iron cauldron", "甑|a steamer", "磨|a stone mill",
        "臼|a stone mortar", "砧|a chopping block", "案|a wooden desk", "席|a woven mat",
        "枕|a porcelain pillow", "被|a folded quilt", "帐|a gauze curtain", "帘|a bamboo blind",
        "屏|a folding screen", "榻|a low couch", "几|a small tea table", "案|a desk",
        "旗|a war banner", "帐|a tent", "辇|an imperial cart", "舆|a carriage",
        "钗|a hairpin", "簪|a jade hairpin", "环|a jade ring", "佩|a jade pendant",
        "裳|a robe", "袍|a long robe", "裘|a fur coat", "冠|a scholar's cap",
        "履|a pair of shoes", "靴|leather boots", "袜|a pair of socks", "巾|a cloth scarf",
        "帐|a bed curtain", "扇|a round fan", "秤|a steelyard scale", "尺|a wooden ruler",
        "斗|a measuring bucket", "斛|a grain measure", "钱|coins", "锭|a silver ingot",
        "玺|an imperial seal", "符|a bamboo tally", "节|a ceremonial baton", "笏|a court tablet",
        "戟|a halberd", "矛|a spear", "盾|a round shield", "甲|plate armour",
        "盔|a helmet", "剑|a straight sword", "刀|a curved sabre", "弓|a wooden bow",
        "箭|arrows", "矢|an arrow", "弩|a crossbow", "鞭|a leather whip",
        "锤|an iron hammer", "斧|an axe", "锯|a saw", "凿|a chisel",
        "铲|a shovel", "锄|a hoe", "犁|a plough", "耙|a rake",
        "镰|a sickle", "磨|a millstone", "绳|a hemp rope", "链|an iron chain",
        "枷|a wooden cangue", "镣|iron shackles", "牢|a prison cell", "狱|a prison",
    ],
    # ------------------------------------------------------------------ 自然与天象
    "nature": [
        "山|a tall mountain", "峰|a sharp peak", "岭|a mountain ridge", "谷|a deep valley",
        "崖|a steep cliff", "洞|a dark cave", "穴|a small burrow", "坑|a pit in the ground",
        "土|a mound of earth", "沙|a dune of sand", "泥|wet mud", "尘|floating dust",
        "水|clear water", "河|a river", "江|a broad river", "湖|a calm lake",
        "海|the open sea", "洋|the vast ocean", "溪|a small brook", "泉|a spring of water",
        "池|a stone pond", "波|a water wave", "浪|a cresting wave", "潮|a rising tide",
        "冰|a block of ice", "霜|white frost", "露|dew drops", "雪|falling snow",
        "火|a burning fire", "焰|a tall flame", "烟|a plume of smoke", "灰|a pile of ash",
        "光|a beam of light", "影|a cast shadow", "日|a bright sun", "月|a crescent moon",
        "星|twinkling stars", "云|a white cloud", "雾|thick fog", "霞|pink sunset clouds",
        "虹|a rainbow", "雷|a lightning bolt", "电|a lightning flash", "风|swirling wind",
        "雨|falling rain", "天|the open sky", "地|the ground plane", "空|empty air",
        "气|a wisp of vapour", "泉|a spring", "渊|a deep abyss", "潭|a deep pool",
        "岛|a small island", "岸|a river bank", "滩|a sandy shore", "洲|a river islet",
        "漠|a desert", "原|an open plain", "野|wild open country", "沼|a swamp",
        "岩|a rocky crag", "矿|a mineral vein", "煤|a lump of coal", "炭|charcoal",
        "泉|a spring", "澜|a big wave", "涛|surging waves", "汐|evening tide",
    ],
    # ------------------------------------------------------------------ 人体
    "body": [
        "人|a simple human figure", "身|a human body", "体|a human body",
        "头|a human head", "首|a human head", "面|a human face", "脸|a human face",
        "目|a pair of eyes", "眼|a single eye", "眉|a pair of eyebrows", "耳|a human ear",
        "鼻|a human nose", "口|an open mouth", "嘴|a mouth", "舌|a tongue",
        "牙|white teeth", "齿|a row of teeth", "唇|a pair of lips", "发|long black hair",
        "须|a long beard", "髯|a long beard", "手|an open hand", "掌|an open palm",
        "指|a pointing finger", "拳|a clenched fist", "臂|a bare arm", "肘|a bent elbow",
        "肩|a shoulder", "背|a human back", "胸|a human chest", "腹|a human belly",
        "腰|a waist", "臂|an arm", "腿|a human leg", "脚|a bare foot", "足|a bare foot",
        "膝|a knee", "踝|an ankle", "皮|a piece of skin", "肤|skin",
        "血|a drop of red blood", "汗|beads of sweat", "泪|a falling teardrop",
        "毛|hair", "骨|a human bone", "尸|a covered corpse",
        "眉|eyebrows", "须|a beard", "鬓|grey temples", "颅|a skull",
    ],
    "organ": [
        "心|a red heart", "肝|a liver", "胆|a gall bladder", "肺|a pair of lungs",
        "脾|a spleen", "肾|kidneys", "肠|intestines", "胃|a stomach",
        "脑|a human brain", "髓|bone marrow", "筋|sinews", "脉|a pulsing vein",
        "喉|a throat", "咽|a throat", "胸|a chest", "膛|a chest",
    ],
    # ------------------------------------------------------------------ 人物
    "person": [
        "民|a group of common people", "众|a crowd of people", "群|a group of people",
        "女|a woman in hanfu", "男|a man", "妇|a woman", "夫|a man",
        "子|a small child", "儿|a small child", "童|a child", "孩|a child",
        "老|an old man with a cane", "翁|an old man", "叟|an old man", "妪|an old woman",
        "父|a father figure", "母|a mother figure", "兄|an elder brother",
        "弟|a younger brother", "姐|an elder sister", "妹|a younger sister",
        "王|a king in robes", "帝|an emperor", "皇|an emperor", "君|a ruler",
        "臣|a court official", "官|an official in a hat", "吏|a clerk",
        "兵|a soldier with a spear", "将|a general with a banner", "卒|a foot soldier",
        "士|a scholar", "儒|a confucian scholar", "僧|a monk", "尼|a nun",
        "道|a daoist priest", "仙|an immortal", "神|a deity", "鬼|a ghost",
        "妖|a demon", "魔|a demon", "怪|a monster", "贼|a thief",
        "盗|a robber", "匪|a bandit", "囚|a prisoner", "奴|a servant",
        "仆|a servant", "婢|a maid", "农|a farmer", "工|a craftsman",
        "商|a merchant", "贾|a merchant", "医|a physician", "师|a teacher",
        "徒|an apprentice", "客|a guest", "主|a host", "友|a friend",
        "敌|an enemy", "仇|a foe", "孤|a lone person", "寡|a widow",
        "幼|a young child", "壮|a strong young man", "妇|a married woman",
        "丐|a beggar", "妓|a courtesan", "伶|an actor", "优|a performer",
        "樵|a woodcutter", "渔|a fisherman", "牧|a shepherd", "猎|a hunter",
        "隐|a hermit", "侠|a swordsman", "侯|a nobleman", "相|a prime minister",
    ],
    # ------------------------------------------------------------------ 动作
    "action": [
        "走|walking legs", "行|walking feet", "跑|running legs", "奔|a sprinting figure",
        "出|stepping out through a doorway", "入|entering through a doorway",
        "放|releasing and letting go", "止|stopping short", "交|crossing over",
        "分|splitting apart", "张|opening wide", "弄|playing with",
        "装|dressing up as", "造|building up", "养|raising and feeding",
        "积|piling up", "存|keeping in store", "舍|giving something up",
        "争|fighting over", "战|fighting a battle with spears", "胜|winning",
        "败|being defeated", "亡|perishing", "屈|bending down", "引|leading along",
        "糊|pasting over", "伤|being wounded", "病|falling ill", "死|dying",
        "居|living in", "归|returning home", "回|turning back", "离|leaving",
        "逢|meeting by chance", "到|arriving", "临|arriving at", "寻|searching for",
        "改|changing", "赏|rewarding", "报|reporting", "传|passing on",
        "引|pulling along", "留|staying behind", "添|adding one more",
        "飞|a flying figure", "跳|a leaping figure", "跃|a leaping figure",
        "坐|a seated figure", "立|a standing figure", "站|a standing figure",
        "卧|a lying figure", "躺|a reclining figure", "跪|a kneeling figure",
        "看|an eye looking", "视|a gaze", "望|looking into the distance",
        "瞻|gazing up at something held aloft", "仰|looking upward",
        "俯|looking downward", "眺|gazing far away",
        "观|observing", "见|a meeting of eyes", "听|a listening ear",
        "闻|smelling", "嗅|sniffing", "言|a speaking mouth", "说|a speaking mouth",
        "语|speech", "话|talking", "呼|shouting", "喊|shouting", "叫|shouting",
        "哭|a crying face", "泣|weeping", "笑|a laughing face", "歌|singing",
        "唱|singing", "舞|dancing", "跳|dancing", "弹|plucking a string",
        "吹|blowing", "吸|inhaling", "吃|eating", "食|eating", "饮|drinking",
        "喝|drinking", "咬|biting", "啃|gnawing", "吞|swallowing", "吐|spitting out",
        "拿|holding an object", "持|holding", "握|gripping", "抓|grabbing",
        "握|grasping", "举|lifting overhead", "抬|lifting", "扛|shouldering",
        "抱|embracing", "拥|hugging", "推|pushing", "拉|pulling", "拖|dragging",
        "扔|throwing", "投|throwing", "掷|hurling", "打|striking", "击|striking",
        "敲|knocking", "拍|patting", "砍|chopping", "切|cutting", "割|slicing",
        "刺|stabbing", "射|shooting an arrow", "杀|a killing blow", "斩|beheading",
        "挖|digging", "掘|digging", "种|planting", "耕|ploughing", "收|harvesting",
        "织|weaving", "缝|sewing", "写|writing with a brush", "画|painting",
        "读书|reading a book", "读|reading", "算|counting on fingers", "数|counting",
        "买|buying", "卖|selling", "借|borrowing", "还|returning", "给|handing over",
        "送|giving", "取|taking", "偷|stealing", "抢|snatching", "夺|seizing",
        "藏|hiding", "躲|hiding", "逃|fleeing", "追|chasing", "赶|catching up",
        "等|waiting", "守|guarding", "护|protecting", "救|rescuing", "扶|supporting",
        "待|waiting patiently",
        "撑|propping up", "盖|covering", "遮|shielding", "掩|covering up",
        "开|opening", "关|closing", "合|closing together", "闭|shutting",
        "拆|dismantling", "修|repairing", "磨|grinding", "炼|smelting",
        "烧|burning", "煮|boiling", "蒸|steaming", "炒|stir-frying",
        "洗|washing", "擦|wiping", "扫|sweeping", "倒|pouring out",
        "添|adding one more", "加|adding", "减|removing", "补|patching",
        "断|breaking apart", "折|snapping", "碎|shattering", "倒|toppling",
        "吃|eating", "睡|sleeping", "醒|waking", "梦|dreaming", "思|thinking",
        "想|thinking", "算|calculating", "猜|guessing", "问|asking",
        "答|answering", "教|teaching", "学|studying", "拜|bowing",
        "贺|celebrating", "祭|making an offering", "拜|worshipping",
        "渡|crossing water", "涉|wading", "游|swimming", "钓|fishing",
        "撒|scattering", "播|sowing", "洒|sprinkling", "浇|pouring water",
        "灌|flooding", "堵|blocking", "塞|plugging", "填|filling in",
        "拔|pulling out", "插|inserting", "钉|nailing", "绑|tying up",
        "系|knotting", "解|untying", "挂|hanging up", "挑|carrying on a pole",
    ],
    "motion": [
        "起|rising up", "落|falling down", "升|ascending", "降|descending",
        "上|moving upward", "下|moving downward", "进|advancing", "退|retreating",
        "来|approaching", "去|departing", "往|going", "返|returning",
        "流|flowing", "滚|rolling", "转|spinning", "旋|whirling",
        "飘|drifting", "浮|floating", "沉|sinking", "漂|floating away",
        "散|scattering", "聚|gathering", "分|splitting", "合|joining",
        "摇|shaking", "晃|swaying", "抖|trembling", "震|shaking violently",
        "倒|falling over", "翻|flipping over", "覆|overturning", "倾|tipping over",
        "溢|overflowing", "漏|leaking", "涌|surging", "喷|spurting",
        "缩|shrinking", "胀|swelling", "裂|cracking", "崩|collapsing",
        "环|encircling", "绕|winding around", "围|surrounding", "穿|passing through",
        "越|crossing over", "过|passing by", "随|following", "逐|pursuing",
    ],
    "emotion": [
        "欢|a joyful face", "兴|an excited face", "恩|a grateful gesture",
        "怀|a nostalgic look", "忍|a clenched enduring expression",
        "痛|a pained face", "怒|an angry red face",
        "喜|a delighted face", "哀|a grieving face",
        "乐|a joyful face", "悲|a sorrowful face", "愁|a worried face",
        "忧|an anxious face", "惊|a shocked face", "恐|a terrified face",
        "惧|a fearful face", "慌|a panicked face", "急|a frantic face",
        "恨|a resentful face", "怨|a bitter face", "忌|a jealous glance",
        "妒|a jealous face", "羞|a blushing face", "愧|a shamefaced look",
        "傲|a proud posture", "谦|a humble bow", "恭|a respectful bow",
        "敬|a respectful gesture", "爱|a warm embrace", "恋|a longing gaze",
        "思|a pensive face", "念|a longing face", "忘|a blank forgetful face",
        "疑|a doubtful face", "信|a trusting handshake", "诚|a sincere gesture",
    ],
    # ------------------------------------------------------------------ 修饰类
    "dimension": [
        "长|long", "短|short", "高|tall", "低|low", "深|deep", "浅|shallow",
        "宽|wide", "窄|narrow", "厚|thick", "薄|thin", "重|heavy", "轻|light",
        "大|large", "小|tiny", "多|many", "少|few", "粗|thick and coarse",
        "细|fine and slender", "直|straight", "曲|bent", "弯|curved",
        "尖|sharply pointed", "圆|perfectly round", "方|square", "扁|flat",
        "密|densely packed", "稀|sparse", "硬|hard", "软|soft",
    ],
    "property": [
        "坚|solid and hard", "暴|violent and fierce", "威|awe-inspiring",
        "妙|wonderfully subtle", "奇|strange and odd", "良|fine and good",
        "残|broken and incomplete", "杂|higgledy-piggledy mixed",
        "微|tiny and faint", "久|long-lasting", "近|near at hand",
        "远|far away", "平|flat and level", "快|fast", "全|whole and complete",
        "精|refined and pure", "贫|utterly destitute", "俗|commonplace",
        "繁|densely crowded", "简|plain and simple", "枯|withered and dry",
        "茂|lush and thriving", "晓|at daybreak", "昏|at dusk", "朝|early morning",
        "暮|late evening", "新|brand new", "旧|worn and old", "老|aged", "幼|very young",
        "强|strong", "弱|frail", "壮|sturdy", "盛|flourishing", "衰|declining",
        "富|wealthy", "穷|destitute", "贵|precious", "贱|worthless",
        "美|beautiful", "丑|ugly", "真|genuine", "假|fake",
        "干|dried out", "净|spotlessly clean", "湿|soaked wet",
        "冷|frozen cold", "热|steaming hot", "暖|warm", "凉|cool",
        "亮|bright", "暗|dim", "明|clear and bright", "清|limpid",
        "浊|muddy", "乱|chaotic", "齐|neatly aligned", "整|orderly",
        "香|fragrant", "臭|foul-smelling", "甜|sweet", "苦|bitter",
        "酸|sour", "辣|spicy hot", "咸|salty", "淡|bland",
        "滑|slippery", "稳|steady", "危|precarious", "安|safe and calm",
        "险|dangerous", "满|filled to the brim", "空|completely empty",
        "虚|hollow", "实|solid and full", "破|broken", "整|intact",
        "同|identical in shape to each other", "异|visibly different from the others",
        "齐|evenly matched", "一致|perfectly aligned",
    ],
    "color": [
        "白|white", "黑|black", "红|red", "黄|yellow",
        "青|blue-green", "蓝|blue", "绿|green", "紫|purple",
        "灰|grey", "金|gold", "银|silver", "朱|vermilion",
        "赤|crimson", "碧|jade-green", "翠|emerald", "苍|pale grey-blue",
        "素|plain white", "彩|multicoloured", "墨|ink-black",
    ],
    "number": [
        "一|exactly one", "二|two", "两|two", "三|three", "四|four",
        "五|five", "六|six", "七|seven", "八|eight", "九|nine", "十|ten",
        "百|one hundred", "千|one thousand", "万|ten thousand", "亿|a hundred million",
        "半|one half", "双|a matching pair", "对|a matching pair", "单|a single one",
        "众|many", "群|a whole group", "数|several", "几|a few", "满|full to the brim",
        "零|zero", "独|only one alone", "孤|a single lonely one", "余|a surplus remainder",
    ],
    "measure": [
        "寸|a tiny inch", "尺|a foot-long ruler", "丈|a ten-foot measure",
        "里|a long league", "斤|a weight", "两|a small weight", "斗|a bushel",
        "石|a large grain measure", "升|a pint measure", "分|a tiny fraction",
        "步|a single footstep", "点|a tiny dot", "刻|a quarter of an hour",
        "载|a whole year", "世|a whole era", "代|a single generation",
        "毫|a hair-thin sliver", "丝|a silk-thin thread", "缕|a thin wisp",
        "滴|a single drop", "粒|a single grain", "颗|a single bead", "片|a thin slice",
        "块|a solid chunk", "团|a rounded ball", "堆|a heaped pile", "层|a stacked layer",
        "重|stacked layers", "叠|folded layers", "束|a tied bundle", "串|a string of",
        "排|a neat row", "列|a line of", "阵|a gust", "股|a strand",
    ],
    "direction": [
        "上|above", "下|below", "左|to the left", "右|to the right",
        "前|in front", "后|behind", "中|in the very centre", "内|inside",
        "外|outside", "里|within", "间|between", "旁|beside",
        "东|east", "南|south", "西|west", "北|north", "边|at the edge",
        "底|at the bottom", "顶|at the very top", "侧|at the side", "背|at the back",
    ],
    "time": [
        "春|spring blossoms", "夏|summer sun", "秋|autumn leaves", "冬|winter snow",
        "日|daytime", "夜|night", "晨|dawn", "暮|dusk", "朝|morning",
        "夕|evening", "年|a year", "月|a month", "时|an hourglass",
        "古|ancient times", "今|the present", "先|earlier", "初|the beginning",
    ],
    "weather": [
        "晴|clear weather", "阴|overcast sky", "雨|rain", "雪|snow", "风|wind",
        "霜|frost", "露|dew", "雹|hail", "雾|fog", "雷|thunder",
    ],
    "sound": [
        "声|sound waves", "响|a loud ring", "鸣|a bird cry", "啼|a cry",
        "吼|a roar", "啸|a whistle", "吟|a chant", "唱|singing",
        "哭|weeping", "笑|laughter", "静|silence", "寂|stillness",
        "喧|a noisy crowd", "噪|noise", "哑|a muted mouth",
    ],
    "state": [
        "满|filled to the brim", "空|completely empty", "虚|hollow",
        "实|solid and full", "破|broken", "残缺|chipped and broken",
        "干|dried out", "湿|soaked wet", "湿|wet", "热|steaming hot",
        "冷|frozen cold", "暖|warm", "凉|cool", "亮|bright",
        "暗|dim", "黑|dark", "明|clear and bright", "清|limpid clear",
        "浊|muddy", "净|spotlessly clean", "脏|filthy", "乱|chaotic mess",
        "齐|neatly aligned", "整|orderly", "密|densely packed", "稀|sparse",
        "厚|thick", "薄|thin", "重|heavy", "轻|light", "硬|hard",
        "软|soft", "脆|brittle", "韧|tough", "滑|slippery",
        "直|straight", "曲|bent", "弯|curved", "尖|pointed",
        "钝|blunt", "圆|round", "方|square", "扁|flat",
        "长|long", "短|short", "高|tall", "低|low", "深|deep",
        "浅|shallow", "宽|wide", "窄|narrow", "大|large", "小|small",
        "多|many", "少|few", "新|brand new", "旧|worn old", "老|aged",
        "幼|young", "壮|sturdy", "弱|frail", "强|strong", "盛|flourishing",
        "衰|declining", "富|wealthy", "穷|destitute", "贵|precious",
        "贱|cheap", "美|beautiful", "丑|ugly", "香|fragrant",
        "臭|foul-smelling", "甜|sweet", "苦|bitter", "酸|sour",
        "辣|spicy hot", "咸|salty", "淡|bland", "真|genuine",
        "假|fake", "正|upright", "歪|crooked", "斜|slanted",
        "稳|steady", "危|precarious", "安|safe and calm", "险|dangerous",
    ],
    "material": [
        "木|wood", "石|stone", "铁|iron", "铜|bronze", "金|gold",
        "银|silver", "玉|jade", "珠|pearl", "玻|glass", "陶|pottery",
        "瓷|porcelain", "布|cloth", "丝|silk", "麻|hemp", "皮|leather",
        "纸|paper", "泥|clay", "沙|sand", "冰|ice", "雪|snow",
        "油|oil", "蜡|wax", "漆|lacquer", "墨|ink",
    ],
    # ------------------------------------------------------------------ 抽象（不入画，仅用于识别"该词难画"）
    "abstract": [
        "义|abstract righteousness", "德|abstract virtue", "仁|abstract benevolence",
        "礼|abstract propriety", "智|abstract wisdom", "信|abstract trust",
        "道|abstract the way", "理|abstract principle", "法|abstract law",
        "名|abstract reputation", "利|abstract profit", "势|abstract momentum",
        "权|abstract power", "政|abstract politics", "治|abstract governance",
        "乱|abstract disorder", "祸|abstract calamity", "福|abstract blessing",
        "命|abstract fate", "运|abstract fortune", "缘|abstract affinity",
        "罪|abstract guilt", "功|abstract merit", "过|abstract fault",
        "是|abstract right", "非|abstract wrong", "善|abstract good",
        "恶|abstract evil", "美|abstract beauty", "丑|abstract ugliness",
        "真|abstract truth", "假|abstract falsehood", "虚|abstract emptiness",
        "实|abstract substance", "情|abstract feeling", "意|abstract intent",
        "志|abstract will", "气|abstract spirit", "神|abstract spirit",
        "魂|abstract soul", "魄|abstract soul", "性|abstract nature",
        "质|abstract quality", "量|abstract quantity", "度|abstract degree",
        "数|abstract number", "势|abstract tendency", "机|abstract opportunity",
        "时|abstract time", "运|abstract luck", "数|abstract destiny",
        "文|abstract writing", "化|abstract transformation", "学|abstract learning",
        "术|abstract art", "业|abstract undertaking", "功|abstract achievement",
        "劳|abstract toil", "苦|abstract suffering", "乐|abstract joy",
        "忧|abstract worry", "思|abstract thought", "虑|abstract anxiety",
        "谋|abstract strategy", "计|abstract scheme", "策|abstract plan",
        "勇|abstract courage", "怯|abstract cowardice", "傲|abstract pride",
        "谦|abstract modesty", "恭|abstract respect", "敬|abstract reverence",
        "孝|abstract filial piety", "忠|abstract loyalty", "节|abstract integrity",
        "操|abstract conduct", "品|abstract character", "行|abstract conduct",
    ],
    "function": [
        "不|(no visual)", "无|(no visual)", "之|(no visual)", "而|(no visual)",
        "其|(no visual)", "所|(no visual)", "以|(no visual)", "于|(no visual)",
        "与|(no visual)", "为|(no visual)", "如|(no visual)", "若|(no visual)",
        "者|(no visual)", "也|(no visual)", "矣|(no visual)", "焉|(no visual)",
        "哉|(no visual)", "乎|(no visual)", "乃|(no visual)", "则|(no visual)",
        "且|(no visual)", "夫|(no visual)", "盖|(no visual)", "唯|(no visual)",
        "是|(no visual)", "非|(no visual)", "有|(no visual)", "在|(no visual)",
        "得|(no visual)", "的|(no visual)", "地|(no visual)", "了|(no visual)",
        "着|(no visual)", "过|(no visual)", "把|(no visual)", "被|(no visual)",
        "将|(no visual)", "给|(no visual)", "对|(no visual)", "从|(no visual)",
        "向|(no visual)", "至|(no visual)", "及|(no visual)", "并|(no visual)",
        "但|(no visual)", "因|(no visual)", "此|(no visual)", "故|(no visual)",
        "何|(no visual)", "莫|(no visual)", "未|(no visual)", "已|(no visual)",
        "皆|(no visual)", "尽|(no visual)", "悉|(no visual)", "咸|(no visual)",
        "亦|(no visual)", "又|(no visual)", "复|(no visual)", "更|(no visual)",
        "最|(no visual)", "甚|(no visual)", "极|(no visual), ", "太|(no visual)",
        "自|(no visual)", "相|(no visual)", "互|(no visual)", "共|(no visual)",
        "同|(no visual)", "各|(no visual)", "每|(no visual)", "凡|(no visual)",
        "诸|(no visual)", "众|(no visual)",
    ],
}

# 反义对：用于识别"反义并置式"构图
ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("大", "小"), ("多", "少"), ("长", "短"), ("高", "低"), ("深", "浅"),
    ("厚", "薄"), ("轻", "重"), ("快", "慢"), ("冷", "热"), ("真", "假"),
    ("是", "非"), ("内", "外"), ("上", "下"), ("前", "后"), ("左", "右"),
    ("东", "西"), ("南", "北"), ("生", "死"), ("新", "旧"), ("强", "弱"),
    ("富", "穷"), ("贵", "贱"), ("美", "丑"), ("善", "恶"), ("黑", "白"),
    ("明", "暗"), ("虚", "实"), ("有", "无"), ("进", "退"), ("得", "失"),
    ("荣", "辱"), ("成", "败"), ("存", "亡"), ("安", "危"), ("动", "静"),
    ("刚", "柔"), ("首", "尾"), ("头", "尾"), ("始", "终"), ("表", "里"),
]

# 方位字：用于识别"方位式"构图
SPATIAL_CHARS = set("上下中内外间旁底顶侧前后左右东西南北边")
# 数量字
NUMBER_CHARS = set("一二两三四五六七八九十百千万亿半双对单满余零")
# 动作字（用于识别"动作式"）
ACTION_CHARS = set("走行跑奔飞跳跃坐立卧看视望瞻仰俯眺观见听闻言说语话呼哭笑吃喝拿持握抓举抬扛抱推拉拖扔投打击敲拍砍切刺射杀挖种耕收织缝写画读算买还送取偷抢夺藏追等守护救撑盖遮掩开关合闭拆修磨烧煮蒸炒洗擦扫倒添加减补断折碎睡醒梦思想猜问答教学拜贺祭渡涉游钓撒播洒浇灌堵塞填拔插钉绑系解挂挑")
# 否定字：会意谜面最难表现"没有/不是"，需要提示画师用空位、断裂、打叉等手法
NEGATION_CHARS = set("不无莫非未勿弗毋罔")
# 高频多义字：在成语里经常取引申义，按字面入画容易答错。命中只扣少量分并打风险标记
POLYSEMY_RISK = set(
    "云风水火花木金玉石天日月底心手目头面口道气神生明高长白空落流色骨血毛步足书文字"
)
# 上下文义项纠偏：(多义字, 同现字集合, 应取的义项 kind)
# 例：「人面兽心」里的"面"该取"脸"，而词典主条目是"一碗面条"（食物类，权重更高）。
# 这类错误会直接让出图画错，所以用"同现字"这个廉价但可靠的信号来纠正。
SENSE_TRIGGERS: list[tuple[str, str, str]] = [
    ("面", "人兽心笑满泪佛容貌首", "body"),
    ("面", "条粉汤碗包饼", "food"),
    ("心", "人兽面胆肝肺肠", "organ"),
    ("手", "举拳指握拍", "body"),
    ("目", "人面眉眼", "body"),
]
# 名词后缀：词尾出现时通常不表意（包子/饺子/馒头/石头）
SUFFIX_CHARS = set("子儿头们性者化")

# 动作字的动词英文形式（第三人称单数）。会意"动作式"谜面需要动词而不是名词短语，
# 例如"画蛇添足"要的是 "paints a snake, and adds extra feet"，而不是 "a hanging scroll painting"。
VERB_EN: dict[str, str] = {
    "走": "walks", "行": "walks", "跑": "runs", "奔": "runs", "飞": "flies",
    "跳": "jumps", "跃": "leaps", "坐": "sits", "立": "stands", "卧": "lies down",
    "看": "looks at", "视": "looks at", "望": "gazes into the distance at",
    "观": "observes", "见": "sees", "听": "listens to", "闻": "smells",
    "言": "speaks", "说": "says", "语": "speaks", "话": "talks", "呼": "shouts",
    "哭": "weeps", "笑": "laughs", "吃": "eats", "喝": "drinks", "拿": "holds",
    "持": "holds", "握": "grips", "抓": "grabs", "举": "lifts high",
    "抬": "lifts", "扛": "shoulders", "抱": "embraces", "推": "pushes",
    "拉": "pulls", "拖": "drags", "扔": "throws away", "投": "throws",
    "打": "hits", "击": "strikes", "敲": "knocks on", "拍": "pats",
    "砍": "chops", "切": "cuts", "刺": "stabs", "射": "shoots an arrow at",
    "杀": "kills", "挖": "digs", "种": "plants", "耕": "ploughs",
    "收": "harvests", "织": "weaves", "缝": "sews", "写": "writes with a brush",
    "画": "paints with a brush", "读": "reads", "算": "counts on fingers",
    "买": "buys", "还": "returns", "送": "gives away", "取": "takes",
    "偷": "steals", "抢": "snatches", "夺": "seizes", "藏": "hides",
    "追": "chases", "等": "waits for", "守": "guards", "护": "protects",
    "救": "rescues", "撑": "props up", "盖": "covers", "遮": "shields",
    "掩": "covers up", "开": "opens", "关": "closes", "合": "joins together",
    "闭": "shuts", "拆": "dismantles", "修": "repairs", "磨": "grinds",
    "烧": "sets fire to", "煮": "boils", "蒸": "steams", "炒": "stir-fries",
    "洗": "washes", "擦": "wipes", "扫": "sweeps", "倒": "pours out",
    "添": "adds an extra", "加": "adds", "减": "removes", "补": "patches",
    "断": "breaks apart", "折": "snaps", "碎": "shatters",
    "睡": "sleeps", "醒": "wakes up", "梦": "dreams of", "思": "ponders",
    "想": "thinks of", "猜": "guesses", "问": "asks", "答": "answers",
    "教": "teaches", "学": "studies", "拜": "bows to", "贺": "congratulates",
    "祭": "makes an offering to", "渡": "crosses", "涉": "wades across",
    "游": "swims in", "钓": "fishes for", "撒": "scatters", "播": "sows",
    "洒": "sprinkles", "浇": "pours water on", "灌": "floods", "堵": "blocks",
    "塞": "plugs", "填": "fills in", "拔": "pulls out", "插": "inserts",
    "钉": "nails down", "绑": "ties up", "系": "ties a knot in", "解": "unties",
    "挂": "hangs up", "挑": "carries on a shoulder pole",
    "瞻": "gazes up at", "仰": "looks up at", "俯": "looks down at",
    "眺": "gazes into the distance at",
    "出": "steps out of", "入": "enters", "放": "releases", "止": "stops",
    "交": "crosses", "分": "splits", "张": "opens wide", "弄": "plays with",
    "装": "dresses up as", "造": "builds", "养": "raises", "积": "piles up",
    "存": "keeps", "舍": "gives up", "争": "fights for", "战": "fights",
    "胜": "defeats", "败": "is defeated by", "亡": "perishes with",
    "屈": "bends down to", "引": "leads", "糊": "pastes over", "伤": "wounds",
    "病": "falls ill from", "死": "dies of", "居": "lives in", "归": "returns to",
    "回": "turns back to", "离": "leaves", "逢": "meets by chance", "到": "arrives at",
    "临": "arrives at", "寻": "searches for", "改": "changes", "赏": "rewards",
    "报": "reports", "传": "passes on", "留": "stays behind in",
    "待": "waits for", "亡": "loses",
}

# 材质字：与器物组合时表"用…做成"（冰箱、雪糕、石桥）
MATERIAL_CHARS = set("冰雪石铁玉木金银铜纸布丝麻皮泥陶瓷玻")
# 材质名词（用于"made entirely of X"，比词典里的短语更干净）
MATERIAL_NOUN = {
    "冰": "ice", "雪": "snow", "石": "stone", "铁": "iron", "玉": "jade",
    "木": "wood", "金": "gold", "银": "silver", "铜": "bronze", "纸": "paper",
    "布": "cloth", "丝": "silk", "麻": "hemp", "皮": "leather", "泥": "clay",
    "陶": "pottery", "瓷": "porcelain", "玻": "glass",
}
# 部位名词（"鸡腿"要的是 leg 而不是 human leg）
BODY_NOUN = {
    "首": "head", "头": "head",
    "腿": "leg", "脚": "foot", "足": "foot", "舌": "tongue",
    "眼": "eye", "目": "eye", "翅": "wing", "掌": "paw", "尾": "tail",
    "骨": "bone", "皮": "skin", "心": "heart", "肝": "liver", "血": "blood",
    "耳": "ear", "鼻": "nose", "嘴": "mouth", "口": "mouth", "牙": "tooth",
    "角": "horn", "爪": "claw", "毛": "fur", "肉": "meat", "蛋": "egg",
}


# 表里/外表-内心 对立对：命中则用"表里式"构图（人面兽心 = 外表是人、内心是兽）
INNER_PAIRS: list[tuple[str, str]] = [
    ("面", "心"), ("貌", "心"), ("表", "里"), ("口", "心"), ("言", "行"),
    ("名", "实"), ("面", "背"), ("头", "心"),
]


class Lexicon:
    """查询接口。"""

    def __init__(self, src: dict[str, list[str]] | None = None):
        self.entries: dict[str, dict[str, Any]] = {}
        self.senses: dict[str, list[dict[str, Any]]] = {}
        self.by_kind: dict[str, list[str]] = {}
        for kind, rows in (src or LEXICON_SRC).items():
            weight, role, mode = KIND_PROPS.get(kind, (0.5, "object", "draw"))
            for row in rows:
                parts = [p.strip() for p in row.split("|")]
                ch = parts[0]
                if not ch:
                    continue
                en = parts[1] if len(parts) > 1 else ch
                alt = parts[2] if len(parts) > 2 else ""
                entry = {
                    "char": ch, "en": en, "alt": alt, "kind": kind,
                    "weight": weight, "role": role, "mode": mode,
                }
                self.senses.setdefault(ch, []).append(entry)
                # 同一字出现在多个类别时，主条目保留权重最高的一条
                prev = self.entries.get(ch)
                if prev and prev["weight"] >= weight:
                    continue
                self.entries[ch] = entry
                self.by_kind.setdefault(kind, []).append(ch)

    # ---------------------------------------------------------------- #
    def sense_in(self, ch: str, kinds: set[str]) -> dict[str, Any] | None:
        """取该字在指定语义类下的义项（用于消歧，如 香醋 的"香"取"芳香的"）。"""
        for e in self.senses.get(ch, []):
            if e["kind"] in kinds:
                return e
        return None

    def resolve_senses(self, text: str, segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按"同现字"纠正多义字的义项。

        只做加法：命中触发条件且该字确实有对应义项时才替换，否则原样返回。
        """
        if not SENSE_TRIGGERS:
            return segs
        chars = set(text)
        out: list[dict[str, Any]] = []
        for s in segs:
            forced_kind = ""
            for ch, triggers, kind in SENSE_TRIGGERS:
                if s["morpheme"] == ch and (chars - {ch}) & set(triggers):
                    forced_kind = kind
                    break
            if forced_kind and forced_kind != s["kind"]:
                alt = self.sense_in(s["morpheme"], {forced_kind})
                if alt:
                    s = dict(s)
                    s.update({
                        "kind": alt["kind"], "en": alt["en"], "role": alt["role"],
                        "mode": alt["mode"], "weight": alt["weight"],
                        "drawable": alt["mode"] in ("draw", "modifier"),
                        "sense_forced": forced_kind,
                    })
            out.append(s)
        return out

    def senses_of(self, ch: str) -> list[str]:
        return [e["kind"] for e in self.senses.get(ch, [])]

    # ---------------------------------------------------------------- #
    def get(self, ch: str) -> dict[str, Any] | None:
        return self.entries.get(ch)

    def is_drawable(self, ch: str) -> bool:
        e = self.entries.get(ch)
        return bool(e) and e["mode"] in ("draw", "modifier")

    def weight(self, ch: str) -> float:
        e = self.entries.get(ch)
        if e:
            return float(e["weight"])
        if ch in FUNCTION_CHARS:
            return 0.0
        return 0.05   # 词典外的字：默认按"大概率抽象"处理

    def coverage(self) -> float:
        """词典对自身条目的覆盖率（自检用）。"""
        return len(self.entries) / max(1, len(self.entries))

    def stats(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.by_kind.items())}

    # ---------------------------------------------------------------- #
    def segment(self, text: str, max_gram: int = 2,
                apply_suffix: bool = False) -> list[dict[str, Any]]:
        """把词条切成"可入画语素 + 功能字"，优先匹配更长的双字词。

        apply_suffix=True 时（用于复合词），词尾的 子/儿/头 等后缀按虚词处理，
        避免"包子"被拆成 包(食物)+子(小孩) 这种伪元素组合。
        返回 [{'morpheme','en','kind','weight','role','mode','drawable'}, ...]
        """
        out: list[dict[str, Any]] = []
        i = 0
        n = len(text)
        while i < n:
            if (apply_suffix and n >= 3 and i == n - 1 and text[i] in SUFFIX_CHARS):
                out.append({
                    "morpheme": text[i], "en": "", "kind": "function",
                    "weight": 0.0, "role": "modifier", "mode": "no", "drawable": False,
                })
                i += 1
                continue
            matched = None
            for gram in range(min(max_gram, n - i), 1, -1):   # 先试双字
                cand = text[i:i + gram]
                if cand in self.entries and self.entries[cand]["mode"] == "draw":
                    matched = (cand, self.entries[cand])
                    break
            if matched is None:
                ch = text[i]
                e = self.entries.get(ch)
                if e is None and ch in FUNCTION_CHARS:
                    e = {"char": ch, "en": "", "alt": "", "kind": "function",
                         "weight": 0.0, "role": "modifier", "mode": "no"}
                elif e is None:
                    e = {"char": ch, "en": "", "alt": "", "kind": "unknown",
                         "weight": 0.05, "role": "modifier", "mode": "weak"}
                matched = (ch, e)
            morpheme, e = matched
            out.append({
                "morpheme": morpheme,
                "en": e.get("en", ""),
                "kind": e.get("kind", "unknown"),
                "weight": float(e.get("weight", 0.05)),
                "role": e.get("role", "object"),
                "mode": e.get("mode", "weak"),
                "drawable": e.get("mode", "weak") in ("draw", "modifier"),
            })
            i += len(morpheme)
        return self.resolve_senses(text, out)

    # ---------------------------------------------------------------- #
    def composition_hint(self, text: str, segs: list[dict[str, Any]]) -> str:
        """根据语素构成推荐构图范式。"""
        drawable = [s for s in segs if s["drawable"]]
        chars = set(text)

        n_spatial = len(chars & SPATIAL_CHARS)
        n_number = len(chars & NUMBER_CHARS)
        n_action = len(chars & ACTION_CHARS)
        n_contrast = sum(1 for a, b in ANTONYM_PAIRS if a in chars and b in chars)
        n_inner = sum(1 for a, b in INNER_PAIRS if a in chars and b in chars)

        # 表里式：外表与内心/表与里 的对立（人面兽心）——比"反义并置"更具体，优先
        if n_inner >= 1 and len(drawable) >= 3:
            return "inner"
        # 数量 + 属性对比（三长两短、七上八下）：数量是主角
        if n_number >= 2 and len(drawable) >= 2:
            return "quantity"
        # 反义并置
        if n_contrast >= 1 and len(drawable) >= 2:
            return "contrast"
        # 方位式：有方位字且至少两个可画元素
        if n_spatial >= 1 and len(drawable) >= 2:
            return "spatial"
        # 数量式
        if n_number >= 1 and len(drawable) >= 2:
            return "quantity"
        # 动作式
        if n_action >= 1 and len(drawable) >= 2:
            return "action"
        # 并列式
        if len(drawable) >= 2:
            return "juxtapose"
        # 具象替代
        return "metaphor"


LEX = Lexicon()


def lexicon_hash() -> str:
    """词典指纹：内容变了就自动触发 select/plan 重算，避免用到缓存的旧切分。"""
    import hashlib
    import json
    blob = json.dumps({k: sorted(v) for k, v in sorted(LEXICON_SRC.items())},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
