"""会意式中文单图猜谜数据集生成 pipeline。

模块划分（按执行顺序）：
    ingest    读取 JSON 成语表/词语表 -> 归一化统一 schema
    select    可图化(会意性)打分与筛选
    plan      生成结构化构图方案 -> 中英提示词
    generate  调用文生图后端批量出图
    qc        质检（文字泄漏 / 技术质量 / 语义与可解性）
    package   打包 JSONL、划分数据集、统计、人工复核页
"""

__version__ = "1.0.0"
