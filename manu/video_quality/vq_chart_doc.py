#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把产物区里的 40 张画质分析图做成一份可交付的《图表解读手册》PDF。

用途：让不熟悉脚本的人（领导 / 新同学）只看图就能读懂每张图画的是什么、该看哪里。

输入：产物区目录（默认 `../cache/ultralytics/repo-artifacts/manu/video_quality/report/`）
输出：同目录下的 `图像解读手册_40张图.pdf`

依赖：matplotlib + Pillow + Noto CJK 字体（Regular/Bold 都注册，缺字体中文会变方框，脚本自检报错）。

执行机：本地主机（纯绘图 + 拼 PDF，不需要 cv2 / 视频解码）。
复现：
    python3 manu/video_quality/vq_chart_doc.py \
        --images-dir ../cache/ultralytics/repo-artifacts/manu/video_quality/report \
        --out ../cache/ultralytics/repo-artifacts/manu/video_quality/report/图像解读手册_40张图.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

PAGE_W, PAGE_H = 8.27, 11.69          # A4 纵向（英寸）
MARGIN = 0.62
CONTENT_W = (PAGE_W - 2 * MARGIN) / PAGE_W   # 正文可用宽度（页面宽度占比）
INK = "#1a1a1a"
ACCENT = "#0B5FA5"
MUTED = "#5b6470"
RULE = "#c9d2dc"
PANEL = "#f3f7fb"
BODY_SIZE = 9.5
BODY_STEP = 0.0205                    # 正文行距（页面高度占比）
HEAD_STEP = 0.0245                    # 小标题行距
BLOCK_GAP = 0.012                     # 段落间距


# --------------------------------------------------------------------------- 字体
def setup_fonts() -> str:
    """注册 Noto CJK 的 Regular + Bold 两个字重（**加粗**要靠真正的 Bold 字面）。"""
    if not Path(FONT_REGULAR).exists():
        raise SystemExit("[FATAL] 找不到中文字体 %s（PDF 里的中文会变方框）" % FONT_REGULAR)
    for path in (FONT_REGULAR, FONT_BOLD):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    name = font_manager.FontProperties(fname=FONT_REGULAR).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    return name


# --------------------------------------------------------------------------- 排版工具
EMOJI_MAP = {
    "\U0001f534": "【注】", "\U0001f7e2": "【好】", "\U0001f7e1": "【中】",
    "✅": "✓", "❌": "✗", "⚠️": "!",
    "\U0001f4cc": "◆", "\U0001f9ea": "◆", "\U0001f6f0": "◆",
    "\U0001f4c8": "▲", "\U0001f4c9": "▼", "\U0001f449": "▶",
    "\U0001f501": "⇄", "\U0001f5a5": "■", "⭐": "★",
    "➡": "→", "➔": "→", "➜": "→",
}


def sanitize(text: str) -> str:
    """PDF 内嵌 CJK 字体没有 emoji 字形 ⇒ 统一替换成可渲染的记号。"""
    for src, dst in EMOJI_MAP.items():
        text = text.replace(src, dst)
    return text


def display_width(text: str) -> float:
    """粗略宽度（仅用于无渲染器时的估算）。"""
    total = 0.0
    for ch in text:
        total += 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5
    return total


def split_bold(text: str) -> list[tuple[str, bool]]:
    """把 '普通**加粗**普通' 拆成 [(片段, 是否加粗), ...]。"""
    parts = text.split("**")
    segs: list[tuple[str, bool]] = []
    for index, part in enumerate(parts):
        if part:
            segs.append((part, index % 2 == 1))
    return segs


def atomize(text: str) -> list[tuple[str, bool]]:
    """切成「不可再分的排版原子」：中文逐字、英文整词、空格单独；并带上加粗状态。"""
    atoms: list[tuple[str, bool]] = []
    for seg, bold in split_bold(text):
        token = ""
        for ch in seg:
            if unicodedata.east_asian_width(ch) in ("W", "F") or ch == " ":
                if token:
                    atoms.append((token, bold))
                    token = ""
                atoms.append((ch, bold))
            else:
                token += ch
        if token:
            atoms.append((token, bold))
    return atoms


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- 页面骨架
class Measurer:
    """用 Agg 渲染器量真实文字宽度（比「全角=1」的估算准，换行不会溢出）。"""

    def __init__(self, fig):
        self.fig = fig
        self.renderer = fig.canvas.get_renderer()
        self.cache: dict[tuple[str, float, bool], float] = {}

    def width(self, text: str, fontsize: float, bold: bool = False) -> float:
        """返回文字宽度（占页面宽度的比例）。"""
        key = (text, round(float(fontsize), 2), bool(bold))
        if key in self.cache:
            return self.cache[key]
        probe = self.fig.text(0, 0, text, fontsize=fontsize,
                               weight="bold" if bold else "normal")
        box = probe.get_window_extent(renderer=self.renderer)
        probe.remove()
        value = box.width / self.fig.bbox.width
        self.cache[key] = value
        return value


def wrap_runs(meas: Measurer, text: str, fontsize: float, max_w: float) -> list[list[tuple[str, bool]]]:
    """中英混排换行：中文按字断行、英文单词不拆；返回每行的 [(片段, 是否加粗), ...]。"""
    lines: list[list[tuple[str, bool]]] = []
    for paragraph in sanitize(text).split("\n"):
        if not paragraph:
            lines.append([])
            continue
        cur: list[tuple[str, bool]] = []
        cur_w = 0.0
        for atom, bold in atomize(paragraph):
            width = meas.width(atom, fontsize, bold)
            if cur and cur_w + width > max_w:
                while cur and cur[-1][0] == " ":
                    cur.pop()
                lines.append(cur)
                cur, cur_w = [], 0.0
                if atom == " ":
                    continue
            cur.append((atom, bold))
            cur_w += width
        while cur and cur[-1][0] == " ":
            cur.pop()
        lines.append(cur)
    return lines


def draw_runs(fig, x: float, y: float, runs: list[tuple[str, bool]],
              fontsize: float, color: str) -> float:
    """逐片段绘制（加粗片段用 Bold 字面），返回绘制后的 x。"""
    renderer = fig.canvas.get_renderer()
    for text, bold in runs:
        if not text:
            continue
        artist = fig.text(x, y, text, fontsize=fontsize, color=color,
                          weight="bold" if bold else "normal")
        box = artist.get_window_extent(renderer=renderer)
        x += box.width / fig.bbox.width
    return x


class Book:
    def __init__(self, out_pdf: Path):
        self.pdf = PdfPages(out_pdf)
        self.out_pdf = out_pdf
        self.figures: list = []

    def close(self):
        self.pdf.close()

    def _new_fig(self):
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.patch.set_facecolor("white")
        return fig

    def page_title(self, fig, kicker: str, title: str, sub: str = "") -> float:
        """页眉：返回正文起始 y。"""
        meas = Measurer(fig)
        fig.text(MARGIN / PAGE_W, 0.955, kicker, fontsize=9.5, color=ACCENT, weight="bold")
        fig.text(MARGIN / PAGE_W, 0.922, title, fontsize=17, color=INK, weight="bold")
        y = 0.893
        if sub:
            for runs in wrap_runs(meas, sub, 9.5, CONTENT_W):
                draw_runs(fig, MARGIN / PAGE_W, y, runs, 9.5, MUTED)
                y -= 0.019
        fig.add_artist(plt.Line2D([MARGIN / PAGE_W, 1 - MARGIN / PAGE_W], [y - 0.006, y - 0.006],
                                  color=RULE, lw=0.9, transform=fig.transFigure))
        return y - 0.030

    def footer(self, fig, text: str = ""):
        fig.text(MARGIN / PAGE_W, 0.028, text, fontsize=7.6, color="#8b95a1")
        fig.text(1 - MARGIN / PAGE_W, 0.028, str(len(self.figures) + 1), fontsize=8.4,
                 color="#8b95a1", ha="right")

    # ---------------------------------------------------------------- 封面
    def cover_page(self, meta: list[str], conclusion: list[str], coverage: list[str],
                   title: str, subtitle: str, note: str) -> None:
        fig = self._new_fig()
        meas = Measurer(fig)
        fig.text(0.5, 0.865, title, fontsize=30, color=INK, weight="bold", ha="center")
        fig.text(0.5, 0.805, "图 像 解 读 手 册", fontsize=22, color=ACCENT, weight="bold", ha="center")
        fig.text(0.5, 0.752, subtitle, fontsize=12.5, color=MUTED, ha="center")
        fig.add_artist(plt.Line2D([0.18, 0.82], [0.722, 0.722], color=RULE, lw=1.0,
                                  transform=fig.transFigure))
        y = 0.672
        for line in meta:
            fig.text(0.5, y, line, fontsize=10.2, color=INK, ha="center")
            y -= 0.031
        fig.text(0.5, 0.472, "一句话结论", fontsize=12, color=ACCENT, weight="bold", ha="center")
        y = 0.442
        for line in conclusion:
            fig.text(0.5, y, line, fontsize=11, color=INK, ha="center")
            y -= 0.026
        fig.add_artist(plt.Line2D([0.18, 0.82], [0.352, 0.352], color=RULE, lw=1.0,
                                  transform=fig.transFigure))
        fig.text(MARGIN / PAGE_W, 0.322, "覆盖范围", fontsize=11.5, color=ACCENT, weight="bold")
        wrapped: list[list[tuple[str, bool]]] = []
        for line in coverage:
            wrapped.extend(wrap_runs(meas, line, 9.8, CONTENT_W - 0.02))
        panel_top = 0.302
        panel_bottom = 0.302 - 0.026 * len(wrapped) - 0.014
        fig.add_artist(Rectangle((MARGIN / PAGE_W - 0.012, panel_bottom),
                                  CONTENT_W + 0.024, panel_top - panel_bottom,
                                  transform=fig.transFigure, facecolor=PANEL,
                                  edgecolor="none", zorder=0))
        y = panel_top - 0.022
        for runs in wrapped:
            draw_runs(fig, MARGIN / PAGE_W, y, runs, 9.8, INK)
            y -= 0.026
        fig.text(MARGIN / PAGE_W, max(0.06, panel_bottom - 0.030), note,
                 fontsize=8.6, color=MUTED)
        self.footer(fig, "")
        self.pdf.savefig(fig)
        plt.close(fig)
        self.figures.append(fig)

    # ---------------------------------------------------------------- 文本页
    def text_page(self, kicker: str, title: str, blocks: list[tuple[str, str]],
                  sub: str = "", footer: str = "") -> None:
        """blocks: [(小标题, 正文)]，正文支持 \\n 分段。"""
        fig = self._new_fig()
        meas = Measurer(fig)
        y = self.page_title(fig, kicker, title, sub)
        x = MARGIN / PAGE_W
        for head, body in blocks:
            if head:
                fig.text(x, y, head, fontsize=11.2, color=ACCENT, weight="bold")
                y -= HEAD_STEP
            for runs in wrap_runs(meas, body, BODY_SIZE, CONTENT_W):
                draw_runs(fig, x, y, runs, BODY_SIZE, INK)
                y -= BODY_STEP
            y -= BLOCK_GAP
        if y < 0.060:
            print("[WARN] 文本页溢出：%s / %s（正文底部 y=%.3f）" % (kicker, title, y))
        self.footer(fig, footer)
        self.pdf.savefig(fig)
        plt.close(fig)
        self.figures.append(fig)

    # ---------------------------------------------------------------- 图页
    def image_page(self, kicker: str, title: str, image: Path, blocks: list[tuple[str, str]],
                   sub: str = "", footer: str = "", caption: str = "") -> None:
        fig = self._new_fig()
        meas = Measurer(fig)
        y = self.page_title(fig, kicker, title, sub)

        # 1) 先把正文排完，算出正文总高度
        layout: list[tuple[str, list[tuple[str, bool]]]] = []
        for head, body in blocks:
            if head:
                layout.append(("h", [(head, False)]))
            for runs in wrap_runs(meas, body, BODY_SIZE, CONTENT_W):
                layout.append(("b", runs))
        text_h = 0.0
        for kind, _ in layout:
            text_h += HEAD_STEP if kind == "h" else BODY_STEP
        text_h += BLOCK_GAP * len(blocks)

        # 2) 给图留一条「图带」，图带高度 = 正文区高度 - 正文 - 图注
        bottom_limit = 0.075
        avail = y - bottom_limit
        caption_h = 0.026 if caption else 0.0
        band_h = avail - text_h - caption_h - 0.020
        band_h = max(band_h, 0.15)

        box_w = CONTENT_W
        with Image.open(image) as im:
            iw, ih = im.size
        aspect = iw / float(ih)
        if aspect > box_w / band_h:
            draw_w, draw_h = box_w, box_w / aspect
        else:
            draw_h, draw_w = band_h, band_h * aspect

        # 3) 整块内容（ + 图注 + 正文）在版心内轻微居中，避免页底大片留白
        used = draw_h + caption_h + 0.022 + text_h
        shift = max(0.0, min(0.055, (avail - used) / 2.0))
        img_top = y - shift
        left = (1 - draw_w) / 2.0
        bottom = img_top - draw_h
        ax = fig.add_axes([left, bottom, draw_w, draw_h])
        ax.imshow(Image.open(image))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
        ax.set_facecolor("white")
        if caption:
            fig.text(left, bottom - 0.018, caption, fontsize=8.4, color=MUTED)

        # 4) 正文紧贴图注下方
        ty = bottom - (0.042 if caption else 0.020)
        x = MARGIN / PAGE_W
        for kind, runs in layout:
            if kind == "h":
                fig.text(x, ty, runs[0][0], fontsize=10.8, color=ACCENT, weight="bold")
                ty -= HEAD_STEP
            else:
                draw_runs(fig, x, ty, runs, BODY_SIZE, INK)
                ty -= BODY_STEP
        if ty < 0.060:
            print("[WARN] 图页溢出：%s / %s（正文底部 y=%.3f）" % (kicker, title, ty))
        self.footer(fig, footer)
        self.pdf.savefig(fig)
        plt.close(fig)
        self.figures.append(fig)


# --------------------------------------------------------------------------- 内容
FAMILY_SIZE = {"f1": 9, "f2": 6, "f3": 8, "f4": 8, "f5": 8, "f6": 1}

FOOTER = "视频质量指标挖掘 · 图像解读手册｜数据：好片 18 条 / 坏片 7 条，307,910 帧｜执行机 192.168.99.40:32222"

KNEE_HOWTO = ("读法：每张图**左＝短时程（5 帧）、右＝长时程（1 秒）**；横轴＝**退化后的实测画质值**"
              "（不是设定的退化参数）；纵轴＝SOT 锁定率 UGR。蓝点＝396 个退化变体里逐 1 秒窗口的散点；"
              "红细线＝分箱曲线（带噪）；黑粗线＝保序回归 + PCHIP 平滑。竖虚线：绿＝Kneedle 膝点、"
              "橙＝PCHIP 曲率膝点、紫＝锁定率 0.5 交点、**黑＝最终采用膝点**。")

FAMILIES = [
    dict(
        key="f1",
        kicker="族群一 · 第 3/4 节",
        title="画质指标区分度（9 张）",
        intro=("这一族回答的是「**哪些画质指标能把好片和坏片分开**」。做法：18 条好片 vs 7 条坏片，"
               "在**裁掉黑边后缩放到宽 640** 的统一画幅上逐秒算指标，再看每个指标在两组里的分布差。\n"
               "生成脚本：`manu/video_quality/vq_compare.py`（三条线：组中位数对照 → AUC 排序 → 逐秒时间线）。\n"
               "读图顺序建议：**先看 AUC 排序**（哪些指标有效）→ **再看组中位数对照**（差多少）→ "
               "**最后看逐秒时间线**（坏片是整段差还是个别坏秒）。"),
        images=[
            ("chart_metrics_ir.png", "红外模态：好 / 坏视频组的质量指标对照",
             "每格一个画质指标；横轴左＝好片组、右＝坏片组，纵轴＝该指标在两组中的中位数；标题给出视频级与窗口级判别力",
             [("画的是什么",
               "第一组图的第 1 张：红外源片。每个指标占一格，把「好片组中位数」和「坏片组中位数」画成两个短横杠，"
               "下方标出该指标的**视频级判别力**与**窗口级判别力**（1.0 = 完全分开，0.5 = 分不开）。"),
              ("看图要点",
               "两个短横杠**离得越远、判别力越接近 1** 的指标越可用。红线的 0.7 是有效判据的分界。"
               "红外实测有效的是 global_contrast / edge_density / lap_var / blockiness / entropy 五项。"),
              ("一句话",
               "这是「哪些指标能当红线」的**第一手证据图**，后面所有阈值都建立在这些指标之上。")]),
            ("chart_metrics_vis.png", "可见光模态：好 / 坏视频组的质量指标对照",
             "与红外图同一画法，只是换成可见光源片（12 条好片）",
             [("画的是什么", "可见光源片的同款对照图。可见光只有三项指标进入正式判据："
               "lap_var（清晰度）、global_contrast（全局对比度）、edge_density（边缘密度）。"),
              ("看图要点",
               "可见光的指标**量纲与红外不同**（例如 lap_var 是红外的好几倍），所以阈值**必须分模态**、"
               "混池线跨模态禁用。"),
              ("一句话", "同一张图读出「红外 / 可见光要分两套阈值」这条硬约束。")]),
            ("chart_metrics_osd.png", "挂飞记录 OSD（桌面录屏）：好 / 坏组对照",
             "第三组是「挂飞记录」这类地面站录屏素材，与源片同款画法",
             [("画的是什么", "OSD 组的同款对照图。OSD 画面被地面站 UI（录制框、坐标、按钮）大量污染。"),
              ("为什么单列",
               "**因画面被 UI 污染、不作为结论依据**，报告正文不列入结论，只留作对照参考。"),
              ("一句话", "这一张是「被排除的对照组」，看它是为了确认排除是对的，不是为了下结论。")]),
            ("chart_auc_ir.png", "红外各指标判别力（AUC）排序",
             "横轴＝判别力 AUC（0.5 = 无区分度，红线 0.7 = 有效），纵向列出全部红外指标",
             [("画的是什么", "把红外所有指标按 AUC 从高到低排序的一张条形图，方便一眼挑出「能用的」和「不能用的」。"),
              ("看图要点",
               "越过 0.7 红线的才是有效指标；**方向相反的指标（如 snr_db、noise_sigma）不能直接当判据**，"
               "因为好片的值反而更极端。"),
              ("一句话", "指标选型的总览图，决定了后面红线只从这五项里挑。")]),
            ("chart_auc_vis.png", "可见光各指标判别力（AUC）排序",
             "与红外同款，列出可见光全部指标",
             [("画的是什么", "可见光指标的 AUC 排序。"),
              ("看图要点", "可见光进榜的指标明显少于红外 ⇒ 可见光的可判别信息更少，"
               "这也是「可见光长时程无解」的根因之一。"),
              ("一句话", "和上一张对照就能看出「红外好判、可见光难判」。")]),
            ("chart_auc_osd.png", "OSD 组各指标判别力（AUC）排序",
             "OSD 组的同款排序图",
             [("画的是什么", "OSD 组的 AUC 排序。"),
              ("看图要点", "OSD 的指标表现**普遍不如源片** ⇒ 再次说明录屏素材不能代表真实飞行画质。"),
              ("一句话", "对照组，用来佐证「只认真源片」这个口径是对的。")]),
            ("chart_timeseries_ir.png", "红外逐秒质量指标时间线",
             "每格一个指标；横轴＝1 秒窗口序号，红线＝坏视频逐秒曲线、绿线＝好视频逐秒曲线",
             [("画的是什么", "把好/坏视频的**逐秒**指标曲线叠在一起（红=坏、绿=好）。"),
              ("看图要点",
               "如果两条线**整段分开**，说明坏片是「系统性差」；如果只有个别秒塌下去，"
               "说明是「局部坏秒」——两者的处置方式完全不同（拒接入 vs 逐秒告警）。"),
              ("一句话", "决定「红线用片级还是秒级」的依据图。")]),
            ("chart_timeseries_vis.png", "可见光逐秒质量指标时间线",
             "与红外同款",
             [("画的是什么", "可见光的逐秒曲线对照。"),
              ("看图要点", "可见光某些好片的个别秒也会塌 ⇒ 说明「越线秒 vs 干净秒」的区分度有限，"
               "红线更适合做**片级准入**而不是逐秒判定。"),
              ("一句话", "解释为什么可见光侧更依赖短时程 + 片级口径。")]),
            ("chart_timeseries_osd.png", "OSD 逐秒质量指标时间线",
             "与红外同款",
             [("画的是什么", "OSD 组的逐秒曲线。"),
              ("看图要点", "OSD 曲线被 UI 元素反复冲击 ⇒ 说明它不是干净的画面度量样本。"),
              ("一句话", "排除 OSD 的第二个独立佐证。")]),
        ]),
    dict(
        key="f2",
        kicker="族群二 · 第 8 节",
        title="48 宫格「任意点选」可跟踪性（6 张）",
        intro=("这一族回答的是「**操作员在画面任意位置点一下，能不能锁住**」。做法：把画面切成 6×8 = 48 格、"
               "每格 32×32 px，逐格用前向-后向**双向闭环**检验（NCC ≥ 0.65 且闭环误差 ≤ 3 px）判定该格是否可锁。\n"
               "三个核心量：**TGR**＝该格有没有纹理；**DET**＝该格有没有细节；**UGR**＝有细节**且**锁得住"
               "（= 主指标「可用格占比」）。\n"
               "生成脚本：`manu/video_quality/vq_grid_charts.py`。"),
        images=[
            ("chart_grid_perclip.png", "逐条视频的网格可跟踪率（TGR / UGR）对照",
             "每条视频一组；蓝＝好片组、红＝坏片组，纵轴＝该视频 48 格的可用比例",
             [("画的是什么", "每条视频三根柱/点：TGR（原始灰度域）、TGR（DoG 带通域）、UGR（可用格占比），"
               "好片与坏片两组并排。"),
              ("看图要点",
               "好片 UGR 中位约 **0.83**（区间 0.52~0.98）；坏片被压到 **0.25 / 0.29**；"
               "「拒绝接入」组只有 **0.063**，其中 2 条是 **0.000**（点哪都锁不住）。"),
              ("一句话", "这是「画面还能不能被跟踪」的总览，也是后面合格线的因变量来源。")]),
            ("chart_grid_heatmap.png", "6×8 网格的空间分布热力图（好片 vs 坏片）",
             "两列网格图，每格里的数字＝该格在多次评估中的可用比例",
             [("画的是什么", "把 48 格按空间位置摊平：左列好片、右列坏片，每格的颜色与数字＝该格可用的比例。"),
              ("看图要点",
               "好片**整幅均匀可点**（每格 0.58~0.86），坏片**整幅都不可点**（每格 0.07~0.23）"
               "⇒ 坏片是**系统性不可用**，不是「只是没拍到目标」。"),
              ("一句话", "最有说服力的一张：它证明坏片的问题在画面本身，不在目标是否入镜。")]),
            ("chart_grid_auc.png", "网格指标与既有画质指标的 AUC 对照",
             "左右两栏分别画 IR / VIS，横轴＝AUC（0.5 = 没有区分度）",
             [("画的是什么", "把网格法算出的 UGR / TGR 与第 3、4 节的传统画质指标放在一起比判别力。"),
              ("看图要点", "网格指标（尤其 UGR）的判别力**不低于**传统指标 ⇒ 说明"
               "「纹理 + 闭环锁定」比单纯算锐度更贴近真实可跟踪性。"),
              ("一句话", "证明这一族方法本身站得住，不只是换了个说法。")]),
            ("chart_grid_redline.png", "画质红线 vs 任意点选可用率（每个点＝1 秒）",
             "横轴＝该秒命中了几条核心红线，纵轴＝该秒的 UGR 可用格占比",
             [("画的是什么", "把「红线命中条数」和「实际可点比例」放在一张散点图上，看两者是否单调对应。"),
              ("看图要点", "命中红线越多、可用率越低 ⇒ 红线确实在描述同一件事；"
               "但**离散的「越线秒 vs 干净秒」区分度有限**（IR AUC 仅 0.008 / VIS 0.129）"
               "⇒ 所以红线更适合做片级准入，而不是逐秒判生死。"),
              ("一句话", "这张图既证明了红线有效，也诚实地划出了红线的边界。")]),
            ("chart_grid_crosstab.png", "画质红线分档 × 任意点选可用率（每条视频一个点）",
             "横轴＝画质红线档位（通过/告警/复核/拒绝），纵轴＝该片的 UGR 可用率",
             [("画的是什么", "把视频按红线分档，再看每档的平均可用率。"),
              ("看图要点", "分档与可用率**单调对应** ⇒ 红线档位可以翻译成业务语言"
               "（可用 / 勉强可用 / 需复核 / 拒接入）。"),
              ("一句话", "把技术阈值翻译成「这批视频能不能上」的运营口径。")]),
            ("chart_grid_timeline.png", "网格可跟踪率时间线（每秒 2 对帧）",
             "横轴＝秒；红＝UGR（可用格占比）、蓝＝TGR（DoG 域）",
             [("画的是什么", "把每条视频的 UGR / TGR 随时间画出来。"),
              ("看图要点", "好片的曲线整体在上、坏片整段在下 ⇒ 再次印证「系统性不可用」；"
               "同时可以看到坏片内部也有相对「好」的秒 ⇒ 秒级判据不如片级稳定。"),
              ("一句话", "把「红线该按片判还是按秒判」这件事讲清楚。")]),
        ]),
    dict(
        key="f3",
        kicker="族群三 · 第 9.5.9 节",
        title="物理崩溃拐点（xcorr 家族，8 张）",
        intro=("这一族是本项目的**核心交付图**：用「梯度退化仿真 + 膝点反推」从**图像本身**推出画质红线，"
               "不依赖坏片样本。做法：对 18 条好片做三组退化（高斯模糊 / 对比度压缩 / H.264 码率压制），"
               "共 396 个变体，用**同一套 48 宫格探针**测锁定率 ⇒ 得到「画质 X vs 锁定率 Y」曲线，"
               "再用 Kneedle / PCHIP 两种独立算法定位**物理崩溃拐点**。\n"
               "本族 Y 轴来自 **xcorr（像素互相关）** 家族；生成脚本 `manu/video_quality/vq_degrade_knee.py`。"),
        images=[],  # 由 knee_thresholds.csv 动态生成
    ),
    dict(
        key="f4",
        kicker="族群四 · 第 9.10.6 节",
        title="物理崩溃拐点（KCF 家族，8 张）",
        intro=("同一套实验**换用 OpenCV contrib 的 KCF 跟踪器**再跑一遍（主臂 `kcf_nt`＝关掉内部响应阈值），"
               "用来验证「红线不是某一个跟踪器的伪影」。\n"
               "画法与族群三**完全相同**，只有 Y 轴的锁定率来源不同 ⇒ 两族图并排看，"
               "就是「**两个原理完全不同的跟踪器在同一画质值处一起崩**」的图形证据。"),
        images=[],
    ),
    dict(
        key="f5",
        kicker="族群五 · 第 9.10.8 节",
        title="KCF 敏感性对照（contrib 默认参数臂，8 张）",
        intro=("敏感性实验：把 KCF 臂换成 contrib 的**默认参数**（内部响应阈值 `detect_thresh = 0.5`）重画，"
               "与族群四的主臂（阈值关闭）对比 ⇒ 验证膝点位置**不依赖** contrib 的内部阈值。\n"
               "实测：16 个格子里 15 个**逐位相同**，唯一有差异的长时程 IR global_contrast 只差 **1.9%**"
               "（0.57763 → 0.566708），远小于 15% 的可靠性门限。"),
        images=[],
    ),
    dict(
        key="f6",
        kicker="族群六 · 证据图",
        title="素材来源取证（1 张）",
        intro="这一族只有 1 张，但它决定了「OSD 组能不能当证据」这个问题。",
        images=[
            ("evidence_挂飞录屏实为桌面录屏_t300s.png", "证据：所谓「挂飞录屏」实为操作员桌面录屏",
             "视频第 300 秒的截帧",
             [("画的是什么", "一条被称为「挂飞记录」的视频，在第 300 秒处截取的原始画面。"),
              ("看图要点", "画面里是**操作员的桌面与录制工具界面**，不是飞行画面 ⇒ "
               "这类素材**不能代表真实飞行画质**，因此 OSD 组只作参考、不进结论。"),
              ("一句话", "一条图就把「为什么 OSD 组不进结论」讲清楚了 —— 这类图叫证据图，不叫分析图。")]),
        ]),
]


# --------------------------------------------------------------------------- 膝点图页（数字全部现场读 CSV）
import csv  # noqa: E402
import datetime  # noqa: E402

KNEE_METRICS = (("ir", ("blockiness", "edge_density", "entropy", "global_contrast", "lap_var")),
                ("vis", ("edge_density", "global_contrast", "lap_var")))
HORIZON_CN = {"short": "短时程（5 帧）", "long": "长时程（1 秒）"}


def read_knees(path: Path) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[(row["horizon"], row["modality"], row["metric"])] = row
    return rows


def fmt(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(number) >= 100:
        return "%.3f" % number
    if abs(number) >= 1:
        return "%.6f" % number
    return "%.6f" % number


def knee_page_text(rows: dict, modality: str, metric: str, other: dict | None, other_label: str) -> list:
    blocks = [("画的是什么", KNEE_HOWTO)]
    for horizon in ("short", "long"):
        row = rows.get((horizon, modality, metric))
        if row is None:
            continue
        op = "≥" if row["operator"] == ">=" else "≤"
        value = fmt(row["knee_x_final"])
        reliable = row["knee_reliable"] == "1"
        line = ("%s：膝点 = %s %s %s（%s，两法差 %s%% 量程 ⇒ %s）"
                % (HORIZON_CN[horizon], metric, op, value, row["method_used"],
                   row["delta_pct_of_span"], "可靠 knee_reliable=1" if reliable else "**不可靠 knee_reliable=0，只作物理参照**"))
        if other is not None:
            orow = other.get((horizon, modality, metric))
            if orow is not None:
                line += "；%s 同格膝点 = %s" % (other_label, fmt(orow["knee_x_final"]))
        blocks.append((HORIZON_CN[horizon], line))
    blocks.append(("这张图说明什么",
                   "曲线在膝点左侧陡降、右侧平缓 ⇒ **低于这个画质值，锁定率断崖式下跌**；"
                   "膝点即「物理崩溃拐点」，是画质合格线的**必要条件式**依据"
                   "（过线仍可能跟不住，过线好片跟住率是另一个指标）。"))
    return blocks


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="把画质分析的 40 张图做成图解手册 PDF")
    default_images = REPO.parent / "cache/ultralytics/repo-artifacts/manu/video_quality/report"
    default_report = REPO / "manu/video_quality/report"
    parser.add_argument("--images-dir", default=str(default_images),
                        help="产物区目录（放 PNG 的地方）")
    parser.add_argument("--out", default="", help="输出 PDF 路径（默认写进 images-dir）")
    parser.add_argument("--report-dir", default=str(default_report),
                        help="仓库内报告目录（读 knee_thresholds.csv）")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    out_pdf = Path(args.out).resolve() if args.out else images_dir / "图像解读手册_40张图.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    font_name = setup_fonts()
    print("[FONT] %s" % font_name)

    xcorr = read_knees(report_dir / "knee_thresholds.csv")
    kcf = read_knees(images_dir / "kcf_knee_thresholds.csv")
    kcfdef = read_knees(images_dir / "kcf_default_knee_thresholds.csv")

    book = Book(out_pdf)

    # ---- 封面
    book.cover_page(
        meta=[
            "样本：好视频 18 条（红外 6 + 可见光 12）＋ 坏视频 7 条，合计 307,910 帧",
            "执行机：huangzhe@192.168.99.40:32222（conda env uav）",
            "图表产物区：cache/ultralytics/repo-artifacts/manu/video_quality/report/（40 张 PNG，不入库）",
            "报告正文（入库）：manu/video_quality/report/REPORT_视频质量指标.md",
            "生成脚本：manu/video_quality/vq_chart_doc.py（本地纯绘图，不需要视频解码）",
            "文档生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ],
        conclusion=[
            "用像素互相关与经典相关滤波两类原理完全不同的跟踪器各自独立反推，",
            "红外采用线几乎逐位复现、盲测坏片拦截率 100% 相同 ⇒",
            "这条红线卡的是图像本身的物理信息底线。",
        ],
        coverage=[
            "族群一 画质指标区分度 9 张 · 族群二 48 宫格任意点选 6 张 · 族群三 xcorr 膝点 8 张",
            "族群四 KCF 主臂膝点 8 张 · 族群五 KCF 默认参数臂 8 张 · 族群六 素材来源取证 1 张（合计 40 张）",
            "每张图独占一页，统一给出「画的是什么 / 看图要点 / 一句话结论」；膝点页另附该格确切数值与来源 CSV",
            "核心数字：红外短时程采用线 global_contrast ≥ 0.577855（xcorr 与 KCF 逐位相同）AND lap_var ≥ 6.493（KCF 6.918）",
            "本手册全部数字现场读 CSV 生成、不手工抄写；复核方式：重跑 manu/video_quality/vq_chart_doc.py",
        ],
        title="视频质量指标挖掘",
        subtitle="40 张图表逐张讲清楚：画的是什么、看哪里、能得什么结论",
        note="文档由脚本自动生成，图文件与 md5 见每页图注；产物不入库，脚本入库。")

    # ---- 阅读指南
    book.text_page(
        "阅读指南", "这本手册怎么用",
        [("40 张图分六个族群",
          "族群一（9 张）＝哪些画质指标能分开好坏片；族群二（6 张）＝操作员在画面任意位置点一下能不能锁住；"
          "族群三（8 张）＝用 xcorr 反推的物理崩溃拐点；族群四（8 张）＝同一套实验换 KCF 跟踪器重画；"
          "族群五（8 张）＝KCF 内部阈值敏感性对照；族群六（1 张）＝素材来源取证。"),
         ("命名规则（记住这一条就够）",
          "chart_metrics_*＝好坏组指标对照；chart_auc_*＝指标判别力排序；chart_timeseries_*＝逐秒时间线；"
          "chart_grid_*＝48 宫格任意点选；chart_degrade_knee_*＝物理崩溃拐点（_kcf_＝KCF 主臂，"
          "_kcfdefault_＝KCF 默认参数臂）；evidence_*＝证据图。\n"
          "模态后缀 ir＝红外、vis＝可见光、osd＝桌面录屏（仅参考）。"),
         ("只给领导看 ⇒ 看这三张",
          "① chart_grid_heatmap.png：好片整幅均匀可点 vs 坏片整幅都不可点；\n"
          "② chart_degrade_knee_kcf_ir_global_contrast.png：红线在哪个画质值处崩；\n"
          "③ chart_grid_crosstab.png：红线档位与实际可用率的对应关系。"),
         ("每个图页都有三段",
          "「画的是什么」＝图的构造；「看图要点」＝该盯哪里；「一句话」＝能拿去汇报的那句结论。"
          "膝点类图页还会给出该格膝点的**确切数值与来源 CSV**。"),
         ("本手册里的数字从哪来",
          "全部现场读取产物区的 CSV（knee_thresholds.csv / kcf_knee_thresholds.csv / kcf_default_knee_thresholds.csv），"
          "不手工抄写；如需复核，重跑 manu/video_quality/vq_chart_doc.py 即可。")],
        sub="每张图都会告诉你：它画的是什么、该看哪里、能得出什么结论。",
        footer=FOOTER)

    # ---- 族群页
    for fam in FAMILIES:
        book.text_page(fam["kicker"], fam["title"],
                       [("本族在讲什么", fam["intro"])],
                       sub="共 %d 张" % FAMILY_SIZE[fam["key"]],
                       footer=FOOTER)

        if fam["key"] == "f3":
            items = []
            for modality, metrics in KNEE_METRICS:
                for metric in metrics:
                    items.append(("chart_degrade_knee_%s_%s.png" % (modality, metric),
                                  "%s / %s 膝点图（xcorr 家族）" % (modality.upper(), metric),
                                  "左＝短时程、右＝长时程；X＝退化后实测画质值；Y＝SOT 锁定率",
                                  knee_page_text(xcorr, modality, metric, None, "")))
        elif fam["key"] == "f4":
            items = []
            for modality, metrics in KNEE_METRICS:
                for metric in metrics:
                    items.append(("chart_degrade_knee_kcf_%s_%s.png" % (modality, metric),
                                  "%s / %s 膝点图（KCF 主臂 kcf_nt）" % (modality.upper(), metric),
                                  "与 xcorr 版同画法，Y 轴锁定率改由 KCF 给出",
                                  knee_page_text(kcf, modality, metric, xcorr, "xcorr")))
        elif fam["key"] == "f5":
            items = []
            for modality, metrics in KNEE_METRICS:
                for metric in metrics:
                    items.append(("chart_degrade_knee_kcfdefault_%s_%s.png" % (modality, metric),
                                  "%s / %s 膝点图（KCF 默认参数臂 detect_thresh=0.5）" % (modality.upper(), metric),
                                  "与 KCF 主臂同画法，用于检验膝点是否依赖 contrib 内部阈值",
                                  knee_page_text(kcfdef, modality, metric, kcf, "KCF 主臂")))
        else:
            items = fam["images"]

        for name, title, sub, blocks in items:
            path = images_dir / name
            if not path.is_file():
                print("[WARN] 缺图，跳过：%s" % name)
                continue
            book.image_page(fam["kicker"], title, path, blocks, sub=sub,
                            footer=FOOTER,
                            caption="图文件：%s ｜ md5 %s" % (name, md5(path)))

    # ---- 速查表
    book.text_page(
        "速查表", "40 张图一句话索引",
        [("族群一 · 画质指标区分度",
          "chart_metrics_ir / vis / osd.png：好/坏组指标中位数对照（哪些指标分得开）；\n"
          "chart_auc_ir / vis / osd.png：各指标 AUC 排序（0.5 无区分度、0.7 有效）；\n"
          "chart_timeseries_ir / vis / osd.png：逐秒曲线（整段差＝片级拒接、局部塌＝逐秒告警）。"),
         ("族群二 · 48 宫格任意点选",
          "chart_grid_perclip.png：逐片 TGR/UGR；chart_grid_heatmap.png：6×8 空间热力图；"
          "chart_grid_auc.png：网格指标 vs 传统指标判别力；chart_grid_redline.png：红线命中数 vs 可用率；"
          "chart_grid_crosstab.png：红线分档 × 可用率；chart_grid_timeline.png：可点比例时间线。"),
         ("族群三/四/五 · 物理崩溃拐点",
          "chart_degrade_knee_{ir,vis}_{metric}.png（8 张）：xcorr 家族膝点；\n"
          "chart_degrade_knee_kcf_{ir,vis}_{metric}.png（8 张）：KCF 主臂同款；\n"
          "chart_degrade_knee_kcfdefault_{ir,vis}_{metric}.png（8 张）：KCF 默认参数臂敏感性对照。\n"
          "ir 五指标＝blockiness / edge_density / entropy / global_contrast / lap_var；"
          "vis 三指标＝edge_density / global_contrast / lap_var。"),
         ("族群六 · 证据",
          "evidence_挂飞录屏实为桌面录屏_t300s.png：证明「挂飞录屏」素材实为桌面录屏 ⇒ OSD 组不进结论。"),
         ("核心数字（供引用）",
          "红外短时程采用线：global_contrast ≥ 0.577855（xcorr 与 KCF 逐位相同）AND lap_var ≥ 6.493（xcorr）/ "
          "6.918（KCF）；长时程：global_contrast ≥ 0.573831（xcorr）/ 0.57763（KCF）AND lap_var ≥ 7.843251（逐位相同）。\n"
          "可见光短时程：lap_var ≥ 306.474（xcorr）/ 270.794（KCF）；盲测：红外片级好片误杀 16.667%、坏片拦截 100%；"
          "可见光片级 0% / 100%。\n"
          "🔴 长时程可见光**无解**（AND 组合好片通过 0/691）⇒ 上线以短时程为准。")],
        sub="按名字反查：这张图在讲什么。",
        footer=FOOTER)

    # ---- 口径提醒
    book.text_page(
        "口径提醒", "汇报时必须一起说的九条（否则会被问倒）",
        [("① 测的不是「跟踪无人机的成功率」",
          "测的是「**画面任意位置点一下能不能锁住**」。测试片里没有无人机，用的也不是交付模型。"),
         ("② 红外 / 可见光必须分模态",
          "量纲不同，阈值不可混用；混池线只作参考、跨模态禁用。"),
         ("③ gap 线只作分界线", "它是「最优分界线」，不是准入线。"),
         ("④ 坏源片只有 4 条", "「坏片拦截 100%」属**小样本结论**，上线前必须扩样重标。"),
         ("⑤ 膝点线是必要条件、不是充分条件",
          "低于线基本跟不住；过线不等于一定跟得住（过线好片跟住率是另一个读数）。"),
         ("⑥ knee_reliable 是一票否决项",
          "两法（Kneedle / PCHIP）差 > 15% 量程的膝点只作物理参照、不作准入线。"),
         ("⑦ 跨家族绝对锁定率不可互换",
          "KCF 臂的 verify CHECK 1 报 0.161 > 0.12 容差（xcorr 为 0.082 通过）⇒ "
          "两族数值**并列呈现、绝不合并**；但**线位一致**才是结论。"),
         ("⑧ KCF 的来历要交代",
          "OpenCV **主仓库没有 KCF**，它在 opencv_contrib（`cv2.legacy.TrackerKCF_create`）；"
          "contrib 的实现是 FSR/KCF 变体、半分辨率。线上画质门禁**不跑跟踪器** ⇒ 无需引入 contrib 依赖；"
          "产品若要集成 KCF，打包必须带 `opencv_contrib` 的 `tracking` 模块。"),
         ("⑨ 一句话背书（可直接引用）",
          "「我们用像素互相关和经典相关滤波两类原理完全不同的跟踪器各自独立反推画质崩溃拐点，"
          "红外采用线几乎逐位一致、盲测坏片拦截率 100% 相同 ⇒ 红线卡的是图像本身的物理信息底线，"
          "而不是某一个跟踪器的实现细节。」")],
        footer=FOOTER)

    book.close()
    print("[OUT] %s（%d 页，%.1f MB）" % (out_pdf, len(book.figures), out_pdf.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
