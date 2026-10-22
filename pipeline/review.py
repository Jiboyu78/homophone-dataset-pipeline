"""人工复核页与联系表。

review.html 是一个自包含的静态页面（不需要服务器、不联网）：
    · 缩略图网格，按构图范式/难度/质检分筛选，关键词搜索
    · 答案默认隐藏，点击或按空格揭晓（方便请人盲测"这张图能不能猜出来"）
    · 键盘 A=通过 R=驳回 S=跳过，J/K 上下移动；决策存 localStorage
    · 一键导出 decisions.json / decisions.csv，下次可再导入继续
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .utils import LOG

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>会意猜谜数据集 · 人工复核</title>
<style>
  :root { --bg:#f7f6f2; --card:#fff; --line:#e2ded4; --ink:#2f2c28; --dim:#8b8478;
          --ok:#2f8f5b; --bad:#c0504d; --warn:#c98a2b; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.55 "Microsoft YaHei",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:10; background:rgba(247,246,242,.96);
           border-bottom:1px solid var(--line); padding:10px 16px; display:flex;
           flex-wrap:wrap; gap:10px; align-items:center; }
  header h1 { font-size:16px; margin:0 12px 0 0; font-weight:600; }
  select,input,button { font:inherit; padding:5px 9px; border:1px solid var(--line);
           border-radius:7px; background:#fff; color:var(--ink); }
  button { cursor:pointer; }
  button.primary { background:var(--ink); color:#fff; border-color:var(--ink); }
  .stats { margin-left:auto; color:var(--dim); font-size:12.5px; }
  main { display:grid; grid-template-columns:repeat(auto-fill,minmax(248px,1fr));
         gap:14px; padding:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:11px;
          overflow:hidden; display:flex; flex-direction:column; }
  .card img { width:100%; aspect-ratio:1/1; object-fit:cover; background:#efece6;
              cursor:pointer; }
  .meta { padding:8px 10px 10px; font-size:12.5px; }
  .answer { font-size:17px; font-weight:700; letter-spacing:1px; }
  .answer.hidden { filter:blur(7px); user-select:none; }
  .row { display:flex; gap:6px; flex-wrap:wrap; align-items:center;
         color:var(--dim); margin-top:3px; }
  .tag { border:1px solid var(--line); border-radius:5px; padding:0 5px; font-size:11.5px; }
  .tag.ok { color:var(--ok); border-color:#bcd9c6; }
  .tag.bad { color:var(--bad); border-color:#e3bdbd; }
  .tag.warn { color:var(--warn); border-color:#e6d2ac; }
  .prompt { color:var(--dim); font-size:11.5px; margin-top:6px;
            display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;
            overflow:hidden; }
  .acts { display:flex; gap:6px; margin-top:8px; }
  .acts button { flex:1; font-size:12px; padding:4px 0; }
  .decided-ok { outline:3px solid var(--ok); }
  .decided-bad { outline:3px solid var(--bad); opacity:.65; }
  .reveal-all .answer.hidden { filter:none; }
  .hidden-decision { display:none; }
</style>
</head>
<body>
<header>
  <h1>会意猜谜数据集 · 人工复核</h1>
  <select id="fComp"><option value="">全部构图范式</option></select>
  <select id="fDiff"><option value="">全部难度</option>
    <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
  </select>
  <select id="fState">
    <option value="">全部状态</option>
    <option value="none">未复核</option>
    <option value="ok">已通过</option>
    <option value="bad">已驳回</option>
  </select>
  <select id="fDeg" title="降级样本：元素只是并排铺开，未做语义合成">
    <option value="">含降级样本</option>
    <option value="only">只看降级样本</option>
    <option value="skip">排除降级样本</option>
  </select>
  <input id="fSearch" placeholder="搜索答案 / 拼音 / 来源分类" style="width:190px">
  <button id="btnReveal">显示答案</button>
  <button id="btnExport" class="primary">导出 decisions.json</button>
  <button id="btnCsv">导出 CSV</button>
  <label class="tag" style="cursor:pointer"><input type="file" id="fileImport"
         accept=".json" class="hidden-decision">导入决策</label>
  <span class="stats" id="stats"></span>
</header>
<main id="grid"></main>
<script>
const DATA = __DATA__;
const LS_KEY = "hv_review_decisions_v1";
let decisions = {};
try { decisions = JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch(e) { decisions = {}; }
let order = DATA.map((_, i) => i);
let cursor = 0;

const grid = document.getElementById("grid");
const statsEl = document.getElementById("stats");

function save() { localStorage.setItem(LS_KEY, JSON.stringify(decisions)); }

function compOptions() {
  const set = new Set(DATA.map(d => d.composition_zh || d.composition));
  const sel = document.getElementById("fComp");
  [...set].sort().forEach(c => { const o = document.createElement("option");
    o.value = c; o.textContent = c; sel.appendChild(o); });
}

function filtered() {
  const c = document.getElementById("fComp").value;
  const d = document.getElementById("fDiff").value;
  const s = document.getElementById("fState").value;
  const dg = (document.getElementById("fDeg") || {}).value || "";
  const q = document.getElementById("fSearch").value.trim().toLowerCase();
  return DATA.filter((r, i) => {
    if (c && (r.composition_zh || r.composition) !== c) return false;
    if (d && String(r.difficulty) !== d) return false;
    if (dg === "only" && !r.degraded) return false;
    if (dg === "skip" && r.degraded) return false;
    const dec = decisions[r.id];
    if (s === "none" && dec) return false;
    if (s === "ok" && dec !== "ok") return false;
    if (s === "bad" && dec !== "bad") return false;
    if (q) {
      const hay = (r.answer + " " + (r.pinyin||"") + " " + (r.category||"") + " " +
                   (r.composition||"")).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function render() {
  const rows = filtered();
  grid.innerHTML = "";
  rows.forEach((r, idx) => {
    const dec = decisions[r.id];
    const card = document.createElement("div");
    card.className = "card" + (dec === "ok" ? " decided-ok" : dec === "bad" ? " decided-bad" : "");
    card.dataset.idx = idx;
    const qc = r.qc || {};
    const sol = qc.solvable === true ? '<span class="tag ok">可解</span>'
              : qc.solvable === false ? '<span class="tag bad">机器猜不出</span>'
              : '<span class="tag">未验可解</span>';
    const tl = qc.text_free === true ? '<span class="tag ok">无文字</span>'
             : qc.text_free === false ? '<span class="tag bad">有文字</span>'
             : '<span class="tag">未验文字</span>';
    const cov = (qc.element_coverage != null)
      ? `<span class="tag">元素 ${Math.round(qc.element_coverage*100)}%</span>` : "";
    const guess = qc.guessed_answer ? `机器猜：${qc.guessed_answer}` : "";
    // 降级样本显著标出：元素只是并排铺开、没做语义合成，复核时要重点看还能不能猜出来
    const deg = r.degraded
      ? `<span class="tag bad" title="${(r.degraded_reason||"").replace(/"/g,"&quot;")}">降级:并列铺开</span>` : "";
    card.innerHTML = `
      <img loading="lazy" src="${r.image}" alt="${r.answer}">
      <div class="meta">
        <div class="answer hidden" data-answer>${r.answer}</div>
        <div class="row">
          <span class="tag">${r.type === "idiom" ? "成语" : "词语"}</span>
          <span class="tag">${r.composition_zh || r.composition}</span>
          <span class="tag">难度 ${r.difficulty}</span>
          <span class="tag">质检 ${qc.score != null ? qc.score.toFixed(2) : "-"}</span>
          ${deg}
        </div>
        <div class="row">${sol}${tl}${cov}</div>
        ${r.degraded && r.degraded_reason ? `<div class="row">降级原因：${r.degraded_reason}</div>` : ""}
        ${r.pinyin ? `<div class="row">${r.pinyin}</div>` : ""}
        ${r.category ? `<div class="row">分类：${r.category}</div>` : ""}
        ${guess ? `<div class="row">${guess}</div>` : ""}
        <div class="prompt" title="${(r.prompt_en||"").replace(/"/g,"&quot;")}">${r.prompt_en||""}</div>
        <div class="acts">
          <button data-act="ok">通过 (A)</button>
          <button data-act="bad">驳回 (R)</button>
          <button data-act="clear">清除</button>
        </div>
      </div>`;
    card.querySelector("img").onclick = () =>
      card.querySelector("[data-answer]").classList.toggle("hidden");
    card.querySelectorAll("button[data-act]").forEach(b => b.onclick = () => {
      const act = b.dataset.act;
      if (act === "clear") delete decisions[r.id]; else decisions[r.id] = act;
      save(); render();
    });
    grid.appendChild(card);
  });
  const done = Object.keys(decisions).length;
  statsEl.textContent = `显示 ${rows.length} / 共 ${DATA.length} 条 · 已复核 ${done} 条`;
}

function exportJson() {
  const blob = new Blob([JSON.stringify(decisions, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
}

function exportCsv() {
  const lines = ["id,answer,decision"];
  DATA.forEach(r => { if (decisions[r.id]) lines.push(`${r.id},${r.answer},${decisions[r.id]}`); });
  const blob = new Blob(["\ufeff" + lines.join("\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "decisions.csv"; a.click();
}

document.getElementById("btnExport").onclick = exportJson;
document.getElementById("btnCsv").onclick = exportCsv;
document.getElementById("btnReveal").onclick = () => {
  document.body.classList.toggle("reveal-all");
  document.querySelectorAll(".answer").forEach(a => a.classList.remove("hidden"));
};
document.getElementById("fileImport").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fr = new FileReader();
  fr.onload = () => { try { decisions = Object.assign(decisions, JSON.parse(fr.result));
    save(); render(); } catch(err) { alert("导入失败：" + err); } };
  fr.readAsText(f);
};
["fComp","fDiff","fState","fDeg"].forEach(id =>
  document.getElementById(id).onchange = render);
document.getElementById("fSearch").oninput = render;

document.addEventListener("keydown", (e) => {
  const rows = filtered(); if (!rows.length) return;
  if (e.key === "j" || e.key === "ArrowDown") cursor = Math.min(cursor+1, rows.length-1);
  else if (e.key === "k" || e.key === "ArrowUp") cursor = Math.max(cursor-1, 0);
  else if (e.key === "a") decisions[rows[cursor].id] = "ok";
  else if (e.key === "r") decisions[rows[cursor].id] = "bad";
  else if (e.key === "s") cursor = Math.min(cursor+1, rows.length-1);
  else if (e.key === " ") { e.preventDefault();
    document.querySelectorAll("[data-answer]").forEach(a => a.classList.remove("hidden")); }
  else return;
  save(); render();
  const el = grid.querySelector(`.card[data-idx="${cursor}"]`);
  if (el) el.scrollIntoView({block:"center", behavior:"smooth"});
});

compOptions();
render();
</script>
</body>
</html>
"""


def write_review(cfg: Config, records: list[dict[str, Any]]) -> Path:
    import json
    slim = []
    for r in records:
        slim.append({
            "id": r["id"], "answer": r["answer"], "pinyin": r.get("pinyin", ""),
            "type": r.get("type", ""), "category": r.get("category", ""),
            "composition": r.get("composition", ""),
            "composition_zh": r.get("composition_zh", ""),
            "difficulty": r.get("difficulty", 3),
            "image": r.get("image", ""), "prompt_en": r.get("prompt_en", ""),
            "layout": r.get("layout", ""),
            "qc": r.get("qc", {}),
        })
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(slim, ensure_ascii=False))
    out = cfg.output_dir / "review.html"
    out.write_text(html, encoding="utf-8")
    LOG.info("人工复核页：%s（%d 条）", out, len(slim))
    return out


def write_contact_sheets(cfg: Config, records: list[dict[str, Any]],
                         per_sheet: int = 48, cell: int = 220) -> list[Path]:
    """按构图范式分组的联系表，方便一眼扫过整批图。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Pillow 不可用，跳过联系表：%s", exc)
        return []

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)
    except OSError:
        font = ImageFont.load_default()

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("composition_zh") or r.get("composition", "other"),
                          []).append(r)

    out_paths: list[Path] = []
    cols = 8
    for comp, rs in groups.items():
        for page in range(0, len(rs), per_sheet):
            chunk = rs[page:page + per_sheet]
            rows = (len(chunk) + cols - 1) // cols
            W = cols * cell
            H = rows * (cell + 20)
            sheet = Image.new("RGB", (W, H), (250, 249, 246))
            d = ImageDraw.Draw(sheet)
            for i, r in enumerate(chunk):
                r_, c_ = divmod(i, cols)
                x, y = c_ * cell, r_ * (cell + 20)
                try:
                    im = Image.open(cfg.output_dir / r["image"]).convert("RGB")
                    im.thumbnail((cell - 8, cell - 8))
                    sheet.paste(im, (x + 4, y + 4))
                except Exception:  # noqa: BLE001
                    d.rectangle([x + 4, y + 4, x + cell - 4, y + cell - 4],
                                outline=(200, 190, 180))
                d.text((x + 5, y + cell - 2), r["answer"], font=font, fill=(50, 50, 55))
            name = f"contact_sheet_{_slug(comp)}_{page // per_sheet + 1}.png"
            p = cfg.output_dir / name
            sheet.save(p)
            out_paths.append(p)
    LOG.info("联系表：%d 张", len(out_paths))
    return out_paths


def _slug(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch == "_") or "misc"
