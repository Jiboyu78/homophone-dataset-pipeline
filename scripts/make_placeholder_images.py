"""测试用：模拟"桌面端生图工具"产出的图片，按任务清单的约定命名。

    python mk_placeholder.py <任务清单 image_tasks.csv> <输出目录> [数量]

真实场景里这些图是你的桌面 agent 生成的；这里只画几个色块占位，
用来验证"导出描述 → 外部生成 → 导回 → 质检打包"这条闭环。
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

csv_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
count = int(sys.argv[3]) if len(sys.argv) > 3 else 3
out_dir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
rnd = random.Random(7)
made = []
for r in rows[:count]:
    name = r["文件名"]
    size = (r.get("尺寸") or "1024x1024").replace("*", "x")
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError:
        w = h = 1024
    img = Image.new("RGB", (w, h), (rnd.randint(230, 250), rnd.randint(230, 245),
                                    rnd.randint(215, 235)))
    d = ImageDraw.Draw(img)
    for _ in range(rnd.randint(2, 4)):
        x0, y0 = rnd.randint(60, w // 2), rnd.randint(60, h // 2)
        x1, y1 = rnd.randint(w // 2, w - 60), rnd.randint(h // 2, h - 60)
        d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                  fill=(rnd.randint(150, 220), rnd.randint(150, 210),
                        rnd.randint(140, 200)), outline=(60, 60, 70), width=5)
    img.save(out_dir / name, "PNG")
    made.append(name)

print(f"已生成 {len(made)} 张占位图到 {out_dir}：")
for m in made:
    print("  " + m)
