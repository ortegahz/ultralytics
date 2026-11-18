#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PointAnnotator — 目标中心点逐帧标注工具（桌面版）。

替代网页版 ``manu/annotate``：标注对象从「检测框」换成「目标中心点」，帧状态从「有/无标签」
换成 ``pos / neg / ign`` 三态，并且人工标注与模型预推坐标**双轨存储**，误操作可无损回退。

用法::

    python point_annotator.py /path/to/videos                 # 指定视频目录
    python point_annotator.py                                 # 弹出目录选择框
    python point_annotator.py /path/to/videos \
        --preannot-dir /path/to/preannot_rc2_tracked          # 导入模型预推点
    python point_annotator.py --self-test                     # 无窗口自检

数据契约见同目录 ``README.md``。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------------------

WINDOW = "PointAnnotator"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov")
STATES = ("pos", "neg", "ign")
DEFAULT_STATE = "ign"

# 三个叠加层的颜色（BGR）。规范给定的是十六进制 RGB，这里换算成 OpenCV 的 BGR 顺序。
COLOR_GHOST = (0xA0, 0xA0, 0xA0)  # #A0A0A0 上一帧确认位置（暗灰十字）
COLOR_PRED = (0xFF, 0xC8, 0x00)  # #00C8FF 当前帧模型预推点（青蓝空心圆）
COLOR_PTS = (0x00, 0xFF, 0x00)  # #00FF00 当前生效标注（亮绿圆环十字准星）
COLOR_ACCENT = (0x00, 0xAA, 0xFF)  # #FFAA00 HUD 提醒
COLOR_HUD = (0xE0, 0xE0, 0xE0)

MAGNIFIER_SIZE = 248  # 放大镜显示边长（px）
MAGNIFIER_SPAN = 31  # 放大镜取样的原图邻域边长（px），248 / 31 == 8 倍
MAGNIFIER_ZOOM = 8

SNAP_RADIUS = 3  # 质心吸附的 7x7 邻域半径
SNAP_BACKGROUND_RADIUS = 7  # 局部背景估计窗口半径

# GDK 修饰键掩码（``window_gtk.cpp`` 把 ``event->state`` 整体左移 16 位塞进键码高位）
GDK_SHIFT = 0x01
GDK_CONTROL = 0x04
GDK_MOD1 = 0x08  # Linux / Windows 上就是 Alt

ADVANCE_MODES = ("prev", "model")
MODE_LABELS = {"prev": "Prev[1]", "model": "Model[2]"}
VIEW_LABELS = ("RAW", "CLAHE", "INFERNO")

HELP_LINES = (
    ("h", "toggle this help     ESC / x  save and quit"),
    ("e / q", "next / prev frame        E / Q  +/- 10 frames"),
    ("w a s d", "nudge active point 1px   W A S D  nudge 5px"),
    ("Space", "play / pause preview"),
    ("1", "adopt previous frame (t-1) points"),
    ("2", "adopt model pred points"),
    ("Tab", "toggle between t-1 and pred"),
    ("n / i", "mark frame NEG / IGN (auto advance)"),
    ("p / c", "add point at cursor / delete active point"),
    ("r / t / v", "advance mode / centroid snap / view filter"),
    ("[ / ]", "previous / next video in the folder"),
    ("mouse", "left click moves the active point (Alt = no snap)"),
)


# --------------------------------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------------------------------

LOG_FILE_NAME = "point_annotator.log"


def log(message: str) -> None:
    """统一日志出口。

    PyInstaller ``--windowed`` 打包后 ``sys.stdout`` 是 ``None``，直接 ``print`` 会抛
    ``AttributeError``，真正的错误反而被吞掉——所以除了 stdout，再写一份到 exe 同目录的日志文件。
    """
    text = str(message)
    stream = sys.stdout
    if stream is not None:
        try:
            print(text)
        except Exception:
            pass
    try:
        base = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()
        with open(base / LOG_FILE_NAME, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------------------------------
# 修饰键读取
# --------------------------------------------------------------------------------------------------


class ModifierReader:
    """读取实时的 Shift / Alt / Ctrl 状态。

    ``cv2.waitKey`` 只给出按键产生的字符，不给修饰键；``cv2.setMouseCallback`` 两者都不给。
    这里两者都需要：``W`` 微调 5 px 而 ``w`` 微调 1 px，按住 ``Alt`` 临时关闭质心吸附。
    实现走 ``ctypes`` 调 Win32 / Xlib——两者都是操作系统自带，不引入任何第三方依赖。

    X11 路线用 ``XQueryKeymap``：它把 32 字节的键盘状态位图填进调用方给的缓冲区。
    比 ``XQueryPointer`` 稳得多——后者有 5 个必写出参，探测脚本一旦传 NULL，
    Xlib 就会往空指针里写而段错误。

    ⚠️ **位图是按 keycode 索引的，不是修饰键掩码**：X11 协议原文
    「Byte N contains the bits for keys 8N to 8N+7」，
    而 X 的 keycode 从 8 开始（本机实测 ``min_keycode = 8``）
    ⇒ **第 0 字节恒为 0x00**，按「bit0=Shift」去读会得到「永远没有修饰键」。
    修饰键的 keycode 由键盘布局决定 ⇒ 用 ``XKeysymToKeycode`` 现场查
    （Shift_L/R、Control_L/R、Alt_L/R）。

    之所以要绕这一圈：Qt 后端的 ``QTest::keyToAscii`` **不带修饰键**，
    ``Shift+w`` 和 ``w`` 回来是同一个字符（实测 ``keyToAscii(0x41) == 97``），
    光看字符大小写永远分不出 ``w`` / ``W``，``W/A/S/D``（5 px）和 ``E/Q``（±10 帧）会静默失效。
    """

    _VK_SHIFT = 0x10
    _VK_CTRL = 0x11
    _VK_ALT = 0x12
    _GDK_SHIFT = GDK_SHIFT
    _GDK_CTRL = GDK_CONTROL
    _GDK_MOD1 = GDK_MOD1
    # X11 keysym（布局无关的符号）→ keycode 由 ``XKeysymToKeycode`` 现场查
    _X11_KEYSYMS = {
        "shift": (0xFFE1, 0xFFE2),  # Shift_L / Shift_R
        "ctrl": (0xFFE3, 0xFFE4),  # Control_L / Control_R
        "alt": (0xFFE9, 0xFFEA),  # Alt_L / Alt_R
    }

    _PROBE = (
        "import ctypes,sys\n"
        "try:\n"
        "    lib=ctypes.CDLL('libX11.so.6')\n"
        "    lib.XOpenDisplay.restype=ctypes.c_void_p\n"
        "    lib.XOpenDisplay.argtypes=[ctypes.c_char_p]\n"
        "    d=lib.XOpenDisplay(None)\n"
        "    if not d: sys.exit(2)\n"
        "    lib.XQueryKeymap.argtypes=[ctypes.c_void_p,ctypes.c_char_p]\n"
        "    lib.XKeysymToKeycode.restype=ctypes.c_ubyte\n"
        "    lib.XKeysymToKeycode.argtypes=[ctypes.c_void_p,ctypes.c_ulong]\n"
        "    buf=ctypes.create_string_buffer(32)\n"
        "    if not lib.XQueryKeymap(d,buf): sys.exit(3)\n"
        "    if not lib.XKeysymToKeycode(d,0xFFE1): sys.exit(5)\n"
        "    sys.exit(0)\n"
        "except Exception:\n"
        "    sys.exit(4)\n"
    )

    def __init__(self) -> None:
        self._mode = ""
        self._lib = None
        self._display = None
        self._keycodes: dict[str, set[int]] = {}
        if sys.platform.startswith("win"):
            try:
                self._lib = ctypes.windll.user32
                self._mode = "win"
            except Exception:
                pass
            return
        if self._probe_x11() and self._setup_x11():
            self._mode = "x11"

    # -- 可用性 ----------------------------------------------------------------------------------

    def _probe_x11(self) -> bool:
        """在子进程里试一次 X11 查询；子进程崩了也不影响本进程。"""
        if getattr(sys, "frozen", False):
            # 冻结版（PyInstaller）里 ``sys.executable`` 就是 exe 自己，
            # ``exe -c "<探针>"`` 会再开一个标注器窗口 ⇒ 直接放弃 X11 通道
            return False
        try:
            result = subprocess.run(
                [sys.executable, "-c", self._PROBE],
                timeout=15,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False
        return result.returncode == 0

    def _setup_x11(self) -> bool:
        try:
            lib = ctypes.CDLL("libX11.so.6")
            lib.XOpenDisplay.restype = ctypes.c_void_p
            lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
            display = lib.XOpenDisplay(None)
            if not display:
                return False
            lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.XKeysymToKeycode.restype = ctypes.c_ubyte
            lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            keycodes: dict[str, set[int]] = {}
            for name, keysyms in self._X11_KEYSYMS.items():
                codes = {int(lib.XKeysymToKeycode(display, keysym)) for keysym in keysyms}
                codes.discard(0)
                if not codes:
                    return False
                keycodes[name] = codes
            self._lib = lib
            self._display = display
            self._keycodes = keycodes
        except Exception:
            return False
        return True

    @property
    def available(self) -> bool:
        return bool(self._mode)

    # -- 查询 ------------------------------------------------------------------------------------

    def _win_key(self, vk: int) -> bool:
        return bool(self._lib.GetAsyncKeyState(vk) & 0x8000)

    def _x11_pressed(self, name: str, raw: bytes) -> bool:
        """按 **keycode** 解释位图，判断某个修饰键当前是否按下。"""
        for code in self._keycodes.get(name, ()):
            if raw[code // 8] >> (code % 8) & 1:
                return True
        return False

    def state(self) -> tuple[bool, bool, bool]:
        """返回 ``(shift, alt, ctrl)``；不可用时全部为 ``False``。"""
        if self._mode == "win":
            return (
                self._win_key(self._VK_SHIFT),
                self._win_key(self._VK_ALT),
                self._win_key(self._VK_CTRL),
            )
        if self._mode == "x11":
            keys = ctypes.create_string_buffer(32)
            if not self._lib.XQueryKeymap(self._display, keys):
                return False, False, False
            raw = keys.raw
            return (
                self._x11_pressed("shift", raw),
                self._x11_pressed("alt", raw),
                self._x11_pressed("ctrl", raw),
            )
        return False, False, False

    def describe(self) -> str:
        """给 HUD 用的可读状态：读到了修饰键，还是只能降级运行。"""
        return {"win": "MOD:WIN32", "x11": "MOD:X11"}.get(self._mode, "MOD:NONE")


# --------------------------------------------------------------------------------------------------
# 标注数据模型
# --------------------------------------------------------------------------------------------------


def norm_points(raw) -> list[list[int]]:
    """把任意来源的点列表规整成 ``[[x, y], ...]``；坏数据一律丢弃而不是抛异常。"""
    points: list[list[int]] = []
    if not isinstance(raw, (list, tuple)):
        return points
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        points.append([int(round(x)), int(round(y))])
    return points


class AnnotationStore:
    """单条视频的标注状态：逐帧记录 + 三态机 + 双轨坐标。

    每条帧记录是 ``{"state": "pos"|"neg"|"ign", "pts": [[x, y], ...],
    "pred_pts": [[x, y], ...], "edited": bool}``：

    * ``pts`` —— 当前生效标注（人工点 / 模型底稿）；
    * ``pred_pts`` —— 原始模型预推点，**只读不改**，误操作后可用 ``2`` 键无损恢复；
    * ``edited`` —— 该帧是否被人动过（自动继承 / 模型底稿都保持 ``False``）。
    """

    def __init__(self, video_name: str = "", width: int = 0, height: int = 0) -> None:
        self.video = video_name
        self.resolution = [int(width), int(height)]
        self.frames: dict[str, dict] = {}

    # -- 帧记录 ----------------------------------------------------------------------------------

    @staticmethod
    def _blank() -> dict:
        return {"state": DEFAULT_STATE, "pts": [], "pred_pts": [], "edited": False}

    def record(self, index: int) -> dict:
        key = str(int(index))
        record = self.frames.get(key)
        if record is None:
            record = self._blank()
            self.frames[key] = record
        return record

    def get(self, index: int) -> dict | None:
        return self.frames.get(str(int(index)))

    # -- 读取 ------------------------------------------------------------------------------------

    def state(self, index: int) -> str:
        record = self.get(index)
        return record["state"] if record else DEFAULT_STATE

    def pts(self, index: int) -> list[list[int]]:
        record = self.get(index)
        return list(record["pts"]) if record else []

    def pred_pts(self, index: int) -> list[list[int]]:
        record = self.get(index)
        return list(record["pred_pts"]) if record else []

    def edited(self, index: int) -> bool:
        record = self.get(index)
        return bool(record["edited"]) if record else False

    # -- 写入 ------------------------------------------------------------------------------------

    def set_pts(self, index: int, points, edited: bool | None = None) -> dict:
        """写入生效标注。点非空 ⇒ 状态 ``pos``；点清空 ⇒ 回到 ``ign``（不是 ``neg``）。

        清空不等于「人工确认过这是硬负样本」，所以不能自动升成 ``neg``——``neg`` 只由 ``n`` 键产生。
        """
        record = self.record(index)
        record["pts"] = norm_points(points)
        if record["pts"]:
            record["state"] = "pos"
        elif record["state"] == "pos":
            record["state"] = DEFAULT_STATE
        if edited is not None:
            record["edited"] = bool(edited)
        return record

    def set_state(self, index: int, state: str, edited: bool | None = None) -> dict:
        record = self.record(index)
        record["state"] = state if state in STATES else DEFAULT_STATE
        if record["state"] != "pos":
            record["pts"] = []
        if edited is not None:
            record["edited"] = bool(edited)
        return record

    def set_pred_pts(self, index: int, points) -> dict:
        record = self.record(index)
        record["pred_pts"] = norm_points(points)
        return record

    def mark_edited(self, index: int, edited: bool = True) -> dict:
        """只改「人工改过」标记，不动点和状态。"""
        record = self.record(index)
        record["edited"] = bool(edited)
        return record

    # -- 序列化 ----------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        frames = {}
        for key in sorted(self.frames, key=lambda item: int(item)):
            record = self.frames[key]
            frames[key] = {
                "state": record["state"],
                "pts": record["pts"],
                "pred_pts": record["pred_pts"],
                "edited": bool(record["edited"]),
            }
        return {"video": self.video, "resolution": list(self.resolution), "frames": frames}

    @classmethod
    def from_dict(cls, data, video_name: str = "", width: int = 0, height: int = 0) -> AnnotationStore:
        store = cls(video_name or "", width, height)
        if not isinstance(data, dict):
            return store
        store.video = video_name or str(data.get("video") or "")
        resolution = data.get("resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                store.resolution = [int(resolution[0]), int(resolution[1])]
            except (TypeError, ValueError):
                pass
        frames = data.get("frames")
        if isinstance(frames, dict):
            for key, value in frames.items():
                if not isinstance(value, dict):
                    continue
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    continue
                record = cls._blank()
                state = str(value.get("state") or DEFAULT_STATE)
                record["state"] = state if state in STATES else DEFAULT_STATE
                record["pts"] = norm_points(value.get("pts"))
                record["pred_pts"] = norm_points(value.get("pred_pts"))
                record["edited"] = bool(value.get("edited", False))
                store.frames[str(index)] = record
        return store


def save_json(path: Path, payload: dict) -> None:
    """原子落盘：先写临时文件再 ``os.replace``，断电/崩溃不会留下半截 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------------------------------
# 模型预推点导入
# --------------------------------------------------------------------------------------------------


class PreAnnotationLoader:
    """从磁盘导入模型预推点，作为标注底稿。

    识别三种落盘布局：

    * **底稿 JSON** —— ``{"video", "resolution", "frames": {"142": {"pts": [...], "pred_pts": [...]}}}``，
      即工具自己的文件格式；
    * **跟踪记录 JSON** —— ``{"sequence", "per_frame": {"52": [{"pos": [x, y], "score": ...}]}}``，
      跟踪档的中心点，坐标在模型原生网格里；
    * **逐帧 YOLO 树** —— ``<root>/<sequence>/<frame:06d>.txt`` 加 ``scores.json`` 边车，
      每行一个 YOLO 框，其中心即跟踪点（归一化坐标 × 原图宽高即原图像素）。

    导入结果是「帧号 → 记录」，记录里除了点，还带着底稿自带的 ``state`` / ``edited``，
    这样把人改过的底稿再导入时，三态机和人工标记不会丢。
    """

    def __init__(
        self,
        roots: list[Path],
        coord_space: str = "auto",
        native_size: tuple[int, int] = (640, 512),
        crop: tuple[int, int, int, int] = (320, 0, 1280, 1024),
        min_score: float = 0.0,
    ) -> None:
        self.roots = [Path(root) for root in roots if root]
        self.coord_space = coord_space
        self.native_width, self.native_height = native_size
        self.crop_left, self.crop_top, self.crop_width, self.crop_height = crop
        self.min_score = float(min_score)
        # 预推点是「原始视频」的坐标系；如果当前放的是 OSD 版视频（画面被裁剪缩放过），
        # 坐标就对不上，这里记一笔让上层提示
        self.matched_via_alias = False

    @staticmethod
    def _base_stem(stem: str) -> str:
        """``VIDEO_x_osd`` → ``VIDEO_x``：OSD 视频与原始序列共用预推点。"""
        if stem.endswith("_osd"):
            return stem[:-4]
        return stem

    def load(self, video_dir: Path, stem: str, width: int, height: int) -> tuple[dict[int, dict], str]:
        """返回 ``({帧号: 记录}, 来源描述)``；找不到任何来源时返回 ``({}, "")``。

        记录形如 ``{"pts": [[x, y], ...], "pred_pts": [...], "state": "pos"|None, "edited": bool}``。
        """
        names = [stem, self._base_stem(stem)]
        self.matched_via_alias = False
        for root in self.roots:
            for position, name in enumerate(names):
                if not name:
                    continue
                json_path = root / f"{name}.json"
                if json_path.is_file():
                    records = self._read_json(json_path, width, height)
                    if records:
                        self.matched_via_alias = position > 0
                        return records, str(json_path)
                tree_path = root / name
                if tree_path.is_dir():
                    records = self._read_tree(tree_path, width, height)
                    if records:
                        self.matched_via_alias = position > 0
                        return records, str(tree_path)
        return {}, ""

    # -- 底稿 / 跟踪记录 JSON --------------------------------------------------------------------

    def _read_json(self, path: Path, width: int, height: int) -> dict[int, dict]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        frames = data.get("frames")
        if isinstance(frames, dict):
            return self._from_draft(frames)
        per_frame = data.get("per_frame")
        if isinstance(per_frame, dict):
            return self._from_tracked(per_frame, width, height)
        return {}

    @staticmethod
    def _from_draft(frames: dict) -> dict[int, dict]:
        """工具自己的底稿格式：点、模型预推点、三态和「人工改过」标记都保留。"""
        out: dict[int, dict] = {}
        for key, record in frames.items():
            if not isinstance(record, dict):
                continue
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            predicted = norm_points(record.get("pred_pts"))
            explicit = norm_points(record.get("pts"))
            edited = bool(record.get("edited", False))
            state = str(record.get("state") or "")
            # 人工编辑过的帧以 ``pts`` 为准；``neg`` 帧即使带点、带模型预推点也不能算成目标
            # （``set_pts`` 会把状态从 ``neg`` 改回 ``pos``，人工确认过的硬负样本就变了正样本）；
            # 其余没动过的帧才回退到模型预推点，当作可用的初始底稿。
            if state == "neg":
                points = []
            elif edited or explicit:
                points = explicit
            else:
                points = predicted
            if not points and state not in STATES:
                continue
            out[index] = {
                "pts": points,
                "pred_pts": predicted or points,
                "state": state if state in STATES else None,
                "edited": edited,
            }
        return out

    def _from_tracked(self, per_frame: dict, width: int, height: int) -> dict[int, dict]:
        raw: dict[int, list] = {}
        for key, entries in per_frame.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            points: list[list[float]] = []
            if not isinstance(entries, (list, tuple)):
                continue
            for entry in entries:
                position = entry.get("pos") if isinstance(entry, dict) else entry
                if not isinstance(position, (list, tuple)) or len(position) < 2:
                    continue
                try:
                    points.append([float(position[0]), float(position[1])])
                except (TypeError, ValueError):
                    continue
            if points:
                raw[index] = points
        if not raw:
            return {}
        space = self.coord_space
        if space == "auto":
            space = "native" if self._looks_native(raw, width, height) else "full"
        return {
            index: {
                "pts": self._to_full(points, space),
                "pred_pts": [],
                "state": None,
                "edited": False,
            }
            for index, points in raw.items()
        }

    def _looks_native(self, raw: dict[int, list], width: int, height: int) -> bool:
        """自动判断跟踪记录用的是原生网格还是原图像素。

        判据是「所有坐标都落在原生网格内，且原图明显更大」。若原图本身就等于原生网格，
        就没法区分，一律按原图处理——宁可坐标大一点也不要整体错位。
        """
        if width and height and (width <= self.native_width or height <= self.native_height):
            return False
        max_x = max(point[0] for points in raw.values() for point in points)
        max_y = max(point[1] for points in raw.values() for point in points)
        return max_x <= self.native_width and max_y <= self.native_height

    def _to_full(self, points: list[list[float]], space: str) -> list[list[int]]:
        if space != "native":
            return norm_points(points)
        scale_x = self.crop_width / float(self.native_width)
        scale_y = self.crop_height / float(self.native_height)
        return norm_points(
            [[self.crop_left + point[0] * scale_x, self.crop_top + point[1] * scale_y] for point in points]
        )

    # -- 逐帧 YOLO 树 ----------------------------------------------------------------------------

    @staticmethod
    def _read_scores(root: Path) -> dict[int, list[float]]:
        """``scores.json`` 边车：``{"000052": [0.391], ...}``，每帧一个置信度列表。

        列表顺序与同帧 ``.txt`` 里的**有效框**顺序一一对应；
        某一帧缺项 ⇒ 该帧不参与置信度过滤（全部保留）。
        """
        path = root / "scores.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[int, list[float]] = {}
        for key, values in data.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(values, (int, float)):
                values = [values]
            if not isinstance(values, (list, tuple)):
                continue
            scores: list[float] = []
            for value in values:
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    scores.append(0.0)
            out[index] = scores
        return out

    def _read_tree(self, root: Path, width: int, height: int) -> dict[int, dict]:
        out: dict[int, dict] = {}
        try:
            entries = sorted(root.glob("*.txt"))
        except OSError:
            return {}
        scores = self._read_scores(root)
        for entry in entries:
            try:
                index = int(entry.stem)
            except ValueError:
                continue
            try:
                lines = entry.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            frame_scores = scores.get(index) or []
            points: list[list[int]] = []
            box_index = 0  # 只对**有效框**计数——空行/坏行不能吃掉一个分数
            for line in lines:
                fields = line.split()
                if len(fields) < 5:
                    continue
                score = frame_scores[box_index] if box_index < len(frame_scores) else None
                box_index += 1
                if score is not None and score < self.min_score:
                    continue
                try:
                    cx = float(fields[1])
                    cy = float(fields[2])
                except ValueError:
                    continue
                if width and height:
                    points.append([cx * width, cy * height])
                else:
                    points.append([cx, cy])
            if points:
                out[index] = {"pts": points, "pred_pts": [], "state": None, "edited": False}
        return out


# --------------------------------------------------------------------------------------------------
# 视频读取
# --------------------------------------------------------------------------------------------------


class VideoReader:
    """随机访问帧读取器。

    ``cap.set(CAP_PROP_POS_FRAMES, n)`` 之后 ``read()`` 落在**精确**的第 n 帧——本项目 mp4 上已用
    逐帧解码做过像素级比对（完全一致），因此这里保留一个 capture，只在必须时 seek。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.count = int(count) if count and count > 0 else 0
        self._cursor = -1
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._cache.clear()

    def frame(self, index: int) -> np.ndarray | None:
        if index < 0:
            return None
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        if self._cursor + 1 == index:
            ok, frame = self.cap.read()
            if ok:
                self._cursor = index
                self._remember(index, frame)
                return frame
            self._cursor = -1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._cursor = index
        self._remember(index, frame)
        return frame

    def probe_limit(self, hint: int) -> int:
        """帧数未知时二分探测真实帧数；返回 ``0`` 表示「探不出上界，不必裁剪」。

        ``CAP_PROP_FRAME_COUNT`` 为 0 的容器没有上界，只能靠「解一帧试试」二分。
        最多 ~13 次解码（1080p 约 1 秒），只在导入底稿时调用一次。
        """
        if self.count > 0 or hint <= 0:
            return 0
        if self.frame(hint) is not None:
            return 0  # 连最后一帧都能解出来 ⇒ 上界不在这些记录里
        low, high = 0, hint - 1
        while low < high:
            mid = (low + high + 1) // 2
            if self.frame(mid) is not None:
                low = mid
            else:
                high = mid - 1
        return low + 1

    def _remember(self, index: int, frame: np.ndarray) -> None:
        self._cache[index] = frame
        while len(self._cache) > 16:
            self._cache.popitem(last=False)


# --------------------------------------------------------------------------------------------------
# 标注器主程序
# --------------------------------------------------------------------------------------------------


class PointAnnotator:
    """中心点标注器：渲染 + 交互 + 增量落盘。"""

    def __init__(
        self,
        video_dir: Path,
        preannot_dirs: list[Path] | None = None,
        coord_space: str = "auto",
        native_size: tuple[int, int] = (640, 512),
        crop: tuple[int, int, int, int] = (320, 0, 1280, 1024),
        headless: bool = False,
        min_score: float = 0.0,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.headless = headless
        self.mods = ModifierReader()
        self.loader = PreAnnotationLoader(
            roots=[self.video_dir] + list(preannot_dirs or []),
            coord_space=coord_space,
            native_size=native_size,
            crop=crop,
            min_score=min_score,
        )

        self.videos: list[Path] = sorted(
            entry for entry in self.video_dir.iterdir() if entry.is_file() and entry.suffix.lower() in VIDEO_SUFFIXES
        )
        if not self.videos:
            raise SystemExit(f"目录里没有视频：{self.video_dir}")

        # 标注目录跟视频目录同级，原始视频目录不被污染
        try:
            (self.video_dir / "annotations").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._cleanup_tmp()

        self.video_index = 0
        self.reader: VideoReader | None = None
        self.store = AnnotationStore()
        self.index = 0
        self.active = -1
        self.mode = ADVANCE_MODES[0]
        self.snap = False
        self.view = 0
        self.show_help = False
        self.playing = False
        self._play_origin = 0.0
        self._play_origin_index = 0
        self.mouse_pos: tuple[float, float] | None = None
        self.mouse_canvas: tuple[float, float] | None = None
        self.running = True
        self.source = ""
        self.canvas_w = 1280
        self.canvas_h = 720
        self.scale = 1.0
        self.offset = (0.0, 0.0)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._filtered_key: tuple[int, int] | None = None
        self._filtered_cache: np.ndarray | None = None

        self._open_video(self.videos[0])

    def _cleanup_tmp(self, max_age_seconds: float = 600.0) -> None:
        """清掉上次异常退出留下的 ``*.tmp`` 残骸（原子落盘的中间文件）。

        **只删 10 分钟前的**：另一个实例正在写盘时，它的临时文件是新鲜的，
        删了会让那次 ``os.replace`` 抛 ``FileNotFoundError`` ⇒ 丢掉一次编辑。
        """
        folder = self.video_dir / "annotations"
        if not folder.is_dir():
            return
        try:
            stale_files = list(folder.glob("*.tmp"))
        except OSError:
            return
        now = time.time()
        for stale in stale_files:
            try:
                if now - stale.stat().st_mtime < max_age_seconds:
                    continue
                stale.unlink()
            except OSError:
                pass

    # -- 视频切换 --------------------------------------------------------------------------------

    @property
    def annotation_path(self) -> Path:
        return self.video_dir / "annotations" / f"{self.videos[self.video_index].stem}.json"

    def _open_video(self, path: Path) -> None:
        if self.reader is not None:
            self.reader.close()
        self.reader = VideoReader(path)
        # 换了视频 ⇒ 帧号会重置，滤镜缓存的键 (帧号, 滤镜) 可能与新视频撞上 ⇒ 必须失效
        self._filtered_key = None
        self._filtered_cache = None
        self.store = AnnotationStore(path.name, self.reader.width, self.reader.height)
        self.source = ""

        stored = self.annotation_path
        if stored.is_file():
            try:
                data = json.loads(stored.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict):
                self.store = AnnotationStore.from_dict(data, path.name, self.reader.width, self.reader.height)
                self.source = str(stored)

        if not self.source:
            records, source = self.loader.load(self.video_dir, path.stem, self.reader.width, self.reader.height)
            total = self.reader.count if self.reader else 0
            if total <= 0 and records and self.reader is not None:
                # 容器没写帧数 ⇒ 没有上界，只能二分探测（最多 ~13 次解码）
                probed = self.reader.probe_limit(max(records))
                if probed:
                    total = probed
            for frame_index, record in records.items():
                if total and frame_index >= total:
                    # 底稿里的帧号超出视频长度：直接丢，绝不写进标注
                    continue
                points = record.get("pts") or []
                predicted = record.get("pred_pts") or points
                state = record.get("state")
                if state in STATES:
                    # 先落状态：``set_state`` 会把非 pos 的帧的点清掉，
                    # 后面的 ``set_pts`` 再写点 ⇒ 顺序反了会把 neg 帧悄悄变成 pos
                    self.store.set_state(frame_index, state)
                self.store.set_pred_pts(frame_index, predicted)
                self.store.set_pts(frame_index, points)
                if record.get("edited"):
                    self.store.mark_edited(frame_index)
            if records:
                self.source = source
                if self.loader.matched_via_alias and not self.headless:
                    log(
                        "[!] 底稿来自原始视频坐标系，而当前视频是 OSD 版："
                        "OSD 画面的相机区是原图裁剪缩放过的，预推点不会落在目标上。"
                        "要按预推点标注，请改用原始视频目录。"
                    )

        self.index = self._resume_index()
        self.active = 0 if self.store.pts(self.index) else -1

    def _resume_index(self) -> int:
        """断点续标：停在第一帧还没被人动过的地方。"""
        total = self.reader.count if self.reader else 0
        for frame_index in range(total):
            if not self.store.edited(frame_index):
                return frame_index
        return max(0, total - 1)

    def _switch_video(self, delta: int) -> None:
        target = self.video_index + delta
        if target < 0 or target >= len(self.videos):
            return
        self._save()
        self.video_index = target
        self._open_video(self.videos[target])
        self.playing = False

    # -- 落盘 ------------------------------------------------------------------------------------

    def _save(self) -> None:
        try:
            save_json(self.annotation_path, self.store.to_dict())
        except OSError as exc:
            log(f"[!] 标注落盘失败：{exc}")

    # -- 导航 ------------------------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return self.reader.count if self.reader else 0

    def _advance(self, delta: int) -> None:
        """按推进策略换帧。

        ``prev`` 模式：下一帧未人工编辑且当前帧有点 ⇒ 继承当前帧的点；
        ``model`` 模式：下一帧未人工编辑 ⇒ 采用该帧的模型预推点。
        两种模式都只写 ``pts``，不动 ``pred_pts``，也不把帧标成 ``edited``——继承链才能继续。
        """
        target = self.index + delta
        if target < 0:
            return
        total = self.frame_count
        if total > 0:
            if target >= total:
                return
        elif self.reader is not None and self.reader.frame(target) is None:
            # 容器没写帧数（``CAP_PROP_FRAME_COUNT`` 返回 0）：解码不到就是到头了
            return
        if delta == 1 and not self.store.edited(target):
            if self.mode == "prev":
                current = self.store.pts(self.index)
                if current:
                    self.store.set_pts(target, current)
            else:
                predicted = self.store.pred_pts(target)
                if predicted:
                    self.store.set_pts(target, predicted)
        self.index = target
        points = self.store.pts(self.index)
        self.active = 0 if points else -1
        self._resync_playback()
        self._save()

    # -- 编辑 ------------------------------------------------------------------------------------

    def _mark(self, state: str) -> None:
        self.store.set_state(self.index, state, edited=True)
        self.active = -1
        self._save()
        self._advance(1)

    def _nudge(self, dx: int, dy: int) -> None:
        points = self.store.pts(self.index)
        if self.active < 0 or self.active >= len(points):
            return
        width = self.reader.width if self.reader else 0
        height = self.reader.height if self.reader else 0
        x = points[self.active][0] + dx
        y = points[self.active][1] + dy
        if width:
            x = max(0, min(x, width - 1))
        if height:
            y = max(0, min(y, height - 1))
        points[self.active] = [x, y]
        self.store.set_pts(self.index, points, edited=True)
        self._save()

    def _adopt_previous(self) -> None:
        points = self.store.pts(self.index - 1) if self.index > 0 else []
        self.store.set_pts(self.index, points, edited=True)
        self.active = 0 if points else -1
        self._save()

    def _adopt_pred(self) -> None:
        points = self.store.pred_pts(self.index)
        self.store.set_pts(self.index, points, edited=True)
        self.active = 0 if points else -1
        self._save()

    def _toggle_source(self) -> None:
        """在「上一帧的点」与「模型预推点」之间来回切换。"""
        previous = self.store.pts(self.index - 1) if self.index > 0 else []
        predicted = self.store.pred_pts(self.index)
        current = self.store.pts(self.index)
        if predicted and current == predicted:
            target = previous
        elif previous and current == previous:
            target = predicted
        else:
            target = predicted if predicted else previous
        self.store.set_pts(self.index, target, edited=True)
        self.active = 0 if target else -1
        self._save()

    def _add_point(self, x: int, y: int) -> None:
        width = self.reader.width if self.reader else 0
        height = self.reader.height if self.reader else 0
        if width:
            x = max(0, min(x, width - 1))
        if height:
            y = max(0, min(y, height - 1))
        points = self.store.pts(self.index)
        points.append([x, y])
        self.store.set_pts(self.index, points, edited=True)
        self.active = len(points) - 1
        self._save()

    def _delete_active(self) -> None:
        points = self.store.pts(self.index)
        if self.active < 0 or self.active >= len(points):
            return
        points.pop(self.active)
        self.store.set_pts(self.index, points, edited=True)
        self.active = (self.active if self.active < len(points) else len(points) - 1) if points else -1
        self._save()

    def _click(self, x: float, y: float, use_snap: bool) -> None:
        width = self.reader.width if self.reader else 0
        height = self.reader.height if self.reader else 0
        point = self._snap_point(x, y) if use_snap else (x, y)
        px = int(round(point[0]))
        py = int(round(point[1]))
        if width:
            px = max(0, min(px, width - 1))
        if height:
            py = max(0, min(py, height - 1))
        points = self.store.pts(self.index)
        if not points:
            points = [[px, py]]
            self.active = 0
        else:
            if self.active < 0 or self.active >= len(points):
                self.active = 0
            points[self.active] = [px, py]
        self.store.set_pts(self.index, points, edited=True)
        self._save()

    def _snap_point(self, x: float, y: float) -> tuple[float, float]:
        """局部背景减除后的亮度质心。"""
        frame = self._current_frame()
        if frame is None:
            return x, y
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        cx = int(max(0, min(int(round(x)), width - 1)))
        cy = int(max(0, min(int(round(y)), height - 1)))
        x0 = max(0, cx - SNAP_RADIUS)
        y0 = max(0, cy - SNAP_RADIUS)
        x1 = min(width, x0 + 2 * SNAP_RADIUS + 1)
        y1 = min(height, y0 + 2 * SNAP_RADIUS + 1)
        window = gray[y0:y1, x0:x1].astype(np.float32)
        if window.size == 0:
            return x, y
        bx0 = max(0, cx - SNAP_BACKGROUND_RADIUS)
        by0 = max(0, cy - SNAP_BACKGROUND_RADIUS)
        bx1 = min(width, bx0 + 2 * SNAP_BACKGROUND_RADIUS + 1)
        by1 = min(height, by0 + 2 * SNAP_BACKGROUND_RADIUS + 1)
        background = float(np.median(gray[by0:by1, bx0:bx1]))
        residual = np.clip(window - background, 0.0, None)
        total = float(residual.sum())
        if total <= 0.0:
            return x, y
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        new_x = float((residual * grid_x).sum() / total)
        new_y = float((residual * grid_y).sum() / total)
        return (
            max(0.0, min(new_x, width - 1.0)),
            max(0.0, min(new_y, height - 1.0)),
        )

    # -- 渲染 ------------------------------------------------------------------------------------

    def _current_frame(self) -> np.ndarray | None:
        if self.reader is None:
            return None
        return self.reader.frame(self.index)

    def _filtered_frame(self) -> np.ndarray | None:
        """当前帧的显示视图。

        CLAHE 在 1080p 上单次约 32 ms，而主循环每 15 ms 就要重绘一次同一帧——
        不缓存的话滤镜视图会直接卡成幻灯片，所以按 ``(帧号, 滤镜)`` 缓存结果。
        """
        key = (self.index, self.view)
        if self._filtered_key == key and self._filtered_cache is not None:
            return self._filtered_cache
        frame = self._current_frame()
        if frame is None:
            self._filtered_key = None
            self._filtered_cache = None
            return None
        if self.view == 0:
            result = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.view == 1:
                result = cv2.cvtColor(self._clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            else:
                result = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        self._filtered_key = key
        self._filtered_cache = result
        return result

    def _layout(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        scale = min(self.canvas_w / float(width), self.canvas_h / float(height))
        self.scale = scale
        self.offset = (
            (self.canvas_w - width * scale) / 2.0,
            (self.canvas_h - height * scale) / 2.0,
        )

    def _to_canvas(self, x: float, y: float) -> tuple[int, int]:
        ox, oy = self.offset
        return int(round(ox + x * self.scale)), int(round(oy + y * self.scale))

    def _to_video(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.offset
        return (x - ox) / self.scale, (y - oy) / self.scale

    @staticmethod
    def _draw_marker(canvas, x: int, y: int, color, size: int, thickness: int, cross: int) -> None:
        cv2.circle(canvas, (x, y), size, color, thickness)
        cv2.line(canvas, (x - cross, y), (x + cross, y), color, 1)
        cv2.line(canvas, (x, y - cross), (x, y + cross), color, 1)

    def _render(self) -> np.ndarray:
        frame = self._filtered_frame()
        if frame is None:
            canvas = np.zeros((self.canvas_h, self.canvas_w, 3), np.uint8)
        else:
            self._layout(frame)
            height, width = frame.shape[:2]
            resized = cv2.resize(
                frame,
                (max(1, int(round(width * self.scale))), max(1, int(round(height * self.scale)))),
            )
            canvas = np.zeros((self.canvas_h, self.canvas_w, 3), np.uint8)
            ox = int(round(self.offset[0]))
            oy = int(round(self.offset[1]))
            canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
            self._draw_overlays(canvas, frame)
        self._draw_hud(canvas)
        if self.show_help:
            self._draw_help(canvas)
        return canvas

    def _draw_overlays(self, canvas: np.ndarray, frame: np.ndarray) -> None:
        # 1) 上一帧确认位置（暗灰十字残影）
        for point in self.store.pts(self.index - 1) if self.index > 0 else []:
            x, y = self._to_canvas(point[0], point[1])
            cv2.line(canvas, (x - 7, y), (x + 7, y), COLOR_GHOST, 1)
            cv2.line(canvas, (x, y - 7), (x, y + 7), COLOR_GHOST, 1)

        # 2) 当前帧模型预推点（青蓝空心圆）
        for point in self.store.pred_pts(self.index):
            x, y = self._to_canvas(point[0], point[1])
            cv2.circle(canvas, (x, y), 9, COLOR_PRED, 1)

        # 3) 当前生效标注（亮绿圆环十字准星），多目标时高亮当前激活点
        for position, point in enumerate(self.store.pts(self.index)):
            x, y = self._to_canvas(point[0], point[1])
            active = position == self.active
            self._draw_marker(
                canvas,
                x,
                y,
                COLOR_PTS,
                10 if active else 8,
                2 if active else 1,
                6 if active else 4,
            )
            if active:
                cv2.circle(canvas, (x, y), 2, COLOR_PTS, -1)

        self._draw_magnifier(canvas, frame)

    def _draw_magnifier(self, canvas: np.ndarray, frame: np.ndarray) -> None:
        focus = self._focus_point()
        if focus is None:
            return
        height, width = frame.shape[:2]
        span = MAGNIFIER_SPAN
        fx = int(max(0, min(int(round(focus[0])), width - 1)))
        fy = int(max(0, min(int(round(focus[1])), height - 1)))
        x0 = max(0, min(fx - span // 2, max(0, width - span)))
        y0 = max(0, min(fy - span // 2, max(0, height - span)))
        x1 = min(width, x0 + span)
        y1 = min(height, y0 + span)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return
        if patch.shape[0] < span or patch.shape[1] < span:
            patch = cv2.copyMakeBorder(
                patch,
                0,
                span - patch.shape[0],
                0,
                span - patch.shape[1],
                cv2.BORDER_REPLICATE,
            )
        magnifier = cv2.resize(patch, (MAGNIFIER_SIZE, MAGNIFIER_SIZE), interpolation=cv2.INTER_NEAREST)
        cross_x = int(round((fx - x0) * MAGNIFIER_SIZE / float(span)))
        cross_y = int(round((fy - y0) * MAGNIFIER_SIZE / float(span)))
        cross_x = max(0, min(cross_x, MAGNIFIER_SIZE - 1))
        cross_y = max(0, min(cross_y, MAGNIFIER_SIZE - 1))
        cv2.line(magnifier, (cross_x, 0), (cross_x, MAGNIFIER_SIZE - 1), (0, 0, 255), 1)
        cv2.line(magnifier, (0, cross_y), (MAGNIFIER_SIZE - 1, cross_y), (0, 0, 255), 1)
        left = self.canvas_w - MAGNIFIER_SIZE - 8
        top = 42
        canvas[top : top + MAGNIFIER_SIZE, left : left + MAGNIFIER_SIZE] = magnifier
        cv2.rectangle(
            canvas,
            (left, top),
            (left + MAGNIFIER_SIZE - 1, top + MAGNIFIER_SIZE - 1),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"{MAGNIFIER_ZOOM}x",
            (left + 4, top + MAGNIFIER_SIZE - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    def _focus_point(self) -> tuple[float, float] | None:
        """放大镜焦点：当前激活标注点 → 当前帧模型预推点 → 鼠标位置。"""
        points = self.store.pts(self.index)
        if 0 <= self.active < len(points):
            return float(points[self.active][0]), float(points[self.active][1])
        predicted = self.store.pred_pts(self.index)
        if predicted:
            return float(predicted[0][0]), float(predicted[0][1])
        if self.mouse_pos is not None:
            return self.mouse_pos
        return None

    def _draw_hud(self, canvas: np.ndarray) -> None:
        height = 34
        strip = canvas[0:height, :]
        canvas[0:height, :] = cv2.addWeighted(strip, 0.35, np.zeros_like(strip), 0.65, 0)
        state = self.store.state(self.index).upper()
        text = (
            f"VIDEO {self.video_index + 1}/{len(self.videos)}"
            f"   FRAME {self.index + 1}/{max(self.frame_count, 1)}"
            f"   STATE {state}"
            f"   MODE {MODE_LABELS[self.mode]}"
            f"   SNAP {'ON' if self.snap else 'OFF'}"
            f"   VIEW {VIEW_LABELS[self.view]}"
            f"   {self.mods.describe()}"
        )
        cv2.putText(canvas, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD, 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        reminder = "Switch to ENG IME"
        (tw, _), _ = cv2.getTextSize(reminder, font, 0.55, 1)
        right = self.canvas_w - tw - 14
        if not self.mods.available:
            # 读不到修饰键 ⇒ W/A/S/D（5 px）与 E/Q（±10 帧）会退化成小写/单帧
            warning = "! NO MODIFIER KEYS"
            (ww, _), _ = cv2.getTextSize(warning, font, 0.55, 1)
            cv2.putText(canvas, warning, (right - ww - 16, 23), font, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, reminder, (right, 23), font, 0.55, COLOR_ACCENT, 1, cv2.LINE_AA)

    def _draw_help(self, canvas: np.ndarray) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        widths = []
        for key, description in HELP_LINES:
            text = f"{key:<10} {description}"
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            widths.append(tw)
        panel_w = max(widths) + 40
        panel_h = len(HELP_LINES) * 22 + 34
        left = max(0, (self.canvas_w - panel_w) // 2)
        top = max(0, (self.canvas_h - panel_h) // 2)
        region = canvas[top : top + panel_h, left : left + panel_w]
        canvas[top : top + panel_h, left : left + panel_w] = cv2.addWeighted(
            region, 0.25, np.zeros_like(region), 0.75, 0
        )
        cv2.rectangle(canvas, (left, top), (left + panel_w - 1, top + panel_h - 1), (120, 120, 120), 1)
        cv2.putText(
            canvas,
            "PointAnnotator - keyboard",
            (left + 20, top + 22),
            font,
            scale,
            COLOR_ACCENT,
            thickness,
            cv2.LINE_AA,
        )
        for row, (key, description) in enumerate(HELP_LINES):
            text = f"{key:<10} {description}"
            cv2.putText(
                canvas,
                text,
                (left + 20, top + 44 + row * 22),
                font,
                scale,
                COLOR_HUD,
                thickness,
                cv2.LINE_AA,
            )

    # -- 事件 ------------------------------------------------------------------------------------

    def _shift_down(self, raw_key: int) -> bool:
        """Shift 判定：优先用权威来源（GTK 键码高位 / 操作系统），都没有才看字符大小写。

        字符大小写只在「完全没有修饰键通道」时兜底（例如受限环境里拿不到 X11 / Win32）：
        Windows 的 ``WM_CHAR`` 带大小写，Caps Lock 开着时会把小写按键误判成 Shift。
        """
        modifiers = self._key_modifiers(raw_key)
        if modifiers is not None:
            return modifiers[0]
        if 0x21 <= raw_key <= 0x7E:
            return chr(raw_key).isupper()
        return False

    def _alt_down(self) -> bool:
        if self.mods.available:
            return self.mods.state()[1]
        return False

    @staticmethod
    def _decode_key(raw_key: int) -> tuple[str, str]:
        """把 ``cv2.waitKey`` 的返回值归一成 ``(kind, char)``。

        OpenCV 各后端的键码约定完全不同（都对着 4.10.0 源码核过）：

        * **Windows**（``window_w32.cpp``）——``WM_CHAR`` 给 ASCII；方向键 / F 键给
          ``VK << 16``，**低 16 位恒为 0**；
        * **Qt**（``window_QT.cpp``）——可翻译键走 ``QTest::keyToAscii``（ASCII，**不带修饰键**，
          所以 ``Shift+w`` 与 ``w`` 同码），方向键走 ``nativeVirtualKey()``（X11 keysym）；
        * **GTK**（``window_gtk.cpp``）——键值是 ``keyval | (event->state << 16)``：
          **低 16 位是键值，高位是 GDK 修饰键掩码**（Shift=0x1 / Ctrl=0x4 / Alt=0x8）。

        这里统一拆开：方向键、GDK 特殊键（``>= 0xff00``）一律挡掉，否则按方向键会误触发微调；
        GTK 的高位修饰键掩码留给 ``_key_modifiers`` 解析。
        """
        if raw_key < 0:
            return "other", ""
        if raw_key >= 0x01000000:  # Qt 原始键码：Esc=0x01000000、Tab=0x01000001
            value = raw_key - 0x01000000
            if value == 0:
                return "esc", ""
            if value == 1:
                return "tab", ""
            return "other", ""
        if raw_key >= 0x10000:
            low = raw_key & 0xFFFF
            if low == 0:
                return "other", ""  # Windows 的 VK<<16 特殊键（方向键 / F 键 / Home / End …）
            if low >= 0xFF00:
                return "other", ""  # GDK / X11 特殊键（方向键 / 功能键 …）
            raw_key = low  # GTK：低位才是真正的键值
        if raw_key == 27:
            return "esc", ""
        if raw_key == 9:
            return "tab", ""
        if raw_key == 32:
            return "space", ""
        if 0x21 <= raw_key <= 0x7E:
            return "char", chr(raw_key)
        return "other", ""

    def _key_modifiers(self, raw_key: int) -> tuple[bool, bool, bool] | None:
        """返回 ``(shift, alt, ctrl)``；返回 ``None`` 表示「没有可靠的修饰键来源」。

        GTK 后端把修饰键掩码编码进键码高位，这种情况下直接用它——它就是按键那一刻的状态；
        其余后端（Windows / Qt）问操作系统。**有权威来源时不要再拿字符大小说事**：
        Windows 的 ``WM_CHAR`` 带大小写，Caps Lock 开着时 ``e`` 会变成 ``E``，
        拿大小写当 Shift 会把「下一帧」误判成「快进 10 帧」。
        """
        if raw_key >= 0x10000 and raw_key & 0xFFFF:
            state = (raw_key >> 16) & 0xFFFF
            return (
                bool(state & GDK_SHIFT),
                bool(state & GDK_MOD1),
                bool(state & GDK_CONTROL),
            )
        if self.mods.available:
            return self.mods.state()
        return None

    def handle_key(self, raw_key: int) -> None:
        """处理一个按键。``raw_key`` 是 ``cv2.waitKey`` 的返回值。"""
        kind, char = self._decode_key(raw_key)
        if kind == "other":
            return
        if kind == "esc":  # ESC -> 保存并退出
            self.quit()
            return
        if kind == "tab":
            self._toggle_source()
            return
        if kind == "space":
            self._toggle_playback()
            return
        lower = char.lower()
        shift = self._shift_down(raw_key)

        if lower == "h":
            self.show_help = not self.show_help
            return
        if lower == "x":  # ESC 的备选退出键，后端吞掉 ESC 时还能退
            self.quit()
            return
        if lower == "r":
            self.mode = ADVANCE_MODES[1] if self.mode == ADVANCE_MODES[0] else ADVANCE_MODES[0]
            return
        if lower == "t":
            self.snap = not self.snap
            return
        if lower == "v":
            self.view = (self.view + 1) % len(VIEW_LABELS)
            return
        if lower == "n":
            self._mark("neg")
            return
        if lower == "i":
            self._mark("ign")
            return
        if lower == "p":
            if self.mouse_pos is not None:
                self._add_point(int(round(self.mouse_pos[0])), int(round(self.mouse_pos[1])))
            return
        if lower == "c":
            self._delete_active()
            return
        if lower == "e":
            self._advance(10 if shift else 1)
            return
        if lower == "q":
            self._advance(-10 if shift else -1)
            return
        if lower == "w":
            self._nudge(0, -5 if shift else -1)
            return
        if lower == "s":
            self._nudge(0, 5 if shift else 1)
            return
        if lower == "a":
            self._nudge(-5 if shift else -1, 0)
            return
        if lower == "d":
            self._nudge(5 if shift else 1, 0)
            return
        if char == "1":
            self._adopt_previous()
            return
        if char == "2":
            self._adopt_pred()
            return
        if char == "[":
            self._switch_video(-1)
            return
        if char == "]":
            self._switch_video(1)
            return

    def handle_mouse(self, event: int, x: int, y: int, flags: int = 0, param=None) -> None:
        """处理一个鼠标事件。

        ``(x, y)`` 已经是**画布像素**，不能按窗口尺寸再缩放一次：OpenCV 各后端在回调之前
        就把光标位置换算成了显示图像的像素坐标。Qt 后端的实现在
        ``modules/highgui/src/window_QT.cpp`` 的 ``DefaultViewPort::icvmouseProcessing``——
        ``mouseCoordinate.rx() = floor(pfx / ratioX)``，其中
        ``ratioX = width() / image2Draw_mat->cols``，与 ``draw2D`` 把图像拉伸铺满视口的画法
        自洽；Win32 / GTK 后端同理直接给图像坐标。

        ``flags`` 是后端填的修饰键位（``CV_EVENT_FLAG_ALTKEY`` 等），用来实现
        「按住 Alt 临时关闭质心吸附」；拿不到时回落到 ``ModifierReader``。

        反例（本机实测）：画布 1280×720 而 Qt 视口只有 1175×661，
        ``cv2.getWindowImageRect`` 返回的也是这个**全局屏幕矩形**，拿它的宽高当缩放因子
        会带来约 9% 的系统性偏差。
        """
        canvas_x = float(x)
        canvas_y = float(y)
        video_x, video_y = self._to_video(canvas_x, canvas_y)
        self.mouse_canvas = (canvas_x, canvas_y)
        self.mouse_pos = (video_x, video_y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Alt 优先用事件自带的后端标志位（各后端都会填 CV_EVENT_FLAG_ALTKEY），
            # 拿不到标志位时才回落到轮询操作系统的修饰键状态。
            alt = bool(flags & cv2.EVENT_FLAG_ALTKEY) or self._alt_down()
            self._click(video_x, video_y, self.snap and not alt)

    def _toggle_playback(self) -> None:
        if self.playing:
            self.playing = False
            return
        if self._at_last_frame():
            # 已经停在片尾：再按空格要从头播，否则一按下就又结束了
            self.index = 0
            self.active = 0 if self.store.pts(0) else -1
        self.playing = True
        self._play_origin = time.monotonic()
        self._play_origin_index = self.index

    def _at_last_frame(self) -> bool:
        """当前是不是最后一帧。帧数未知（容器没写）时靠「下一帧解不出来」判断。"""
        total = self.frame_count
        if total > 0:
            return self.index >= total - 1
        if self.reader is None:
            return True
        return self.reader.frame(self.index + 1) is None

    def _resync_playback(self) -> None:
        """手动换帧后把播放时钟挪到当前帧，否则播放会把用户拉回旧位置。"""
        if self.playing:
            self._play_origin = time.monotonic()
            self._play_origin_index = self.index

    def quit(self) -> None:
        self._save()
        self.running = False

    # -- 主循环 ----------------------------------------------------------------------------------

    def _open_window(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.canvas_w, self.canvas_h)
        cv2.setMouseCallback(WINDOW, self.handle_mouse)

    def run(self) -> None:
        if self.headless:
            return
        self._open_window()
        while self.running:
            delay = 15
            if self.playing:
                elapsed = time.monotonic() - self._play_origin
                fps = self.reader.fps if self.reader else 25.0
                total = self.frame_count
                target = self._play_origin_index + int(elapsed * fps)
                if total > 0 and target >= total:
                    self.playing = False
                    target = total - 1
                if target != self.index:
                    if total <= 0 and self.reader is not None and self.reader.frame(target) is None:
                        # 帧数未知：解码失败即片尾
                        self.playing = False
                        target = self.index
                    else:
                        self.index = max(0, target)
                        points = self.store.pts(self.index)
                        self.active = 0 if points else -1
                if self.playing:
                    # 睡到下一帧该显示的时刻，别用 delay=1 忙等烧 CPU
                    next_due = self._play_origin + (self.index - self._play_origin_index + 1) / fps
                    delay = max(1, min(200, int((next_due - time.monotonic()) * 1000)))
            canvas = self._render()
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(delay)
            if key >= 0:
                self.handle_key(key)
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------
# 命令行入口
# --------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PointAnnotator — 目标中心点逐帧标注工具")
    parser.add_argument("video_dir", nargs="?", default="", help="视频目录；留空则弹出图形化目录选择框")
    parser.add_argument(
        "--preannot-dir",
        action="append",
        default=[],
        help="模型预推点目录，可重复；支持逐帧 YOLO 树、底稿 JSON、跟踪记录 JSON",
    )
    parser.add_argument(
        "--coord-space",
        choices=("auto", "full", "native"),
        default="auto",
        help="跟踪记录 JSON 的坐标空间；auto 表示按坐标范围自动判断",
    )
    parser.add_argument("--min-score", type=float, default=0.0, help="导入预推点时的置信度下限（scores.json 边车）")
    parser.add_argument("--native-size", default="640,512", help="模型原生网格尺寸 W,H（native 坐标空间用）")
    parser.add_argument("--crop", default="320,0,1280,1024", help="裁剪契约 left,top,width,height（native 用）")
    parser.add_argument("--self-test", action="store_true", help="运行无窗口自检后退出")
    return parser.parse_args(argv)


def _parse_ints(text: str, count: int) -> tuple[int, ...]:
    parts = [item.strip() for item in str(text).replace(" ", ",").split(",") if item.strip()]
    if len(parts) != count:
        raise SystemExit(f"需要 {count} 个整数，收到：{text}")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"非法整数：{text}") from exc


def ask_directory(initial: str = "") -> str:
    """弹出目录选择框；没有图形环境时退回命令行输入。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="选择视频目录", initialdir=initial or os.getcwd())
        root.destroy()
        return chosen or ""
    except Exception as exc:
        log(f"[!] 无法打开目录选择框（{exc}），请在命令行传入视频目录路径。")
        try:
            return input("视频目录：").strip()
        except EOFError:
            return ""


def screen_size() -> tuple[int, int]:
    """屏幕尺寸，用于给窗口选一个不超出屏幕的画布。"""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1920, 1080


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    video_dir = args.video_dir
    if not video_dir:
        video_dir = ask_directory()
        if not video_dir:
            log("[!] 没有选择目录，退出。")
            return 1
    root = Path(video_dir).expanduser()
    if not root.is_dir():
        log(f"[!] 目录不存在：{root}")
        return 1
    root = root.resolve()

    width, height = screen_size()
    app = PointAnnotator(
        root,
        preannot_dirs=[Path(item) for item in args.preannot_dir],
        coord_space=args.coord_space,
        native_size=_parse_ints(args.native_size, 2),
        crop=_parse_ints(args.crop, 4),
        min_score=args.min_score,
    )
    app.canvas_w = max(960, min(1600, width - 80))
    app.canvas_h = max(540, min(900, height - 140))
    log(f"[i] 视频目录：{root}")
    log(f"[i] 视频数量：{len(app.videos)}")
    log(f"[i] 标注文件：{app.annotation_path}")
    log(f"[i] 修饰键通道：{app.mods.describe()}")
    if app.source:
        log(f"[i] 已载入底稿：{app.source}")
    app.run()
    return 0


# --------------------------------------------------------------------------------------------------
# 无窗口自检
# --------------------------------------------------------------------------------------------------


def _write_test_video(path: Path, frames: int = 40, width: int = 320, height: int = 240) -> Path:
    """造一段合成视频：一个亮斑匀速横移，外加一块会呼吸的背景方块。"""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (width, height),
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(path.with_suffix(".avi")),
            cv2.VideoWriter_fourcc(*"MJPG"),
            25.0,
            (width, height),
        )
        path = path.with_suffix(".avi")
    for index in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        cv2.rectangle(frame, (10, 10), (60, 50), (40, 40, 40), -1)
        cv2.circle(frame, (20 + index * 6, 150), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class _SelfTest:
    """收集断言并统一汇报。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, condition: bool, label: str) -> None:
        if condition:
            self.passed += 1
        else:
            self.failed.append(label)

    def equal(self, actual, expected, label: str) -> None:
        self.check(actual == expected, f"{label}: got {actual!r}, expected {expected!r}")


def run_self_test() -> int:
    """跑完规范里点名的边界条件，返回进程退出码。"""
    suite = _SelfTest()
    with tempfile.TemporaryDirectory(prefix="point_annotator_selftest_") as temp:
        root = Path(temp)
        video = _write_test_video(root / "VIDEO00001_test.mp4")
        app = PointAnnotator(root, headless=True)
        suite.equal(app.frame_count > 0, True, "视频帧数可读")

        # -- 边界条件 1：第 0 帧按 1 键 / q 键都不抛异常，且不越界 -------------------------------
        app.index = 0
        app.handle_key(ord("1"))
        suite.equal(app.index, 0, "第 0 帧按 1 键后仍在第 0 帧")
        app.handle_key(ord("q"))
        suite.equal(app.index, 0, "第 0 帧按 q 键不倒退")
        app.handle_key(ord("2"))
        suite.equal(app.index, 0, "第 0 帧按 2 键不越界")

        # -- 边界条件 2：末帧按 e 键不越界 ------------------------------------------------------
        app.index = app.frame_count - 1
        app.handle_key(ord("e"))
        suite.equal(app.index, app.frame_count - 1, "末帧按 e 键不前进")
        app.handle_key(ord("E"))
        suite.equal(app.index, app.frame_count - 1, "末帧按 E 键不前进")

        # -- 边界条件 3：无预推数据时按 2 键不炸，且不产生点 ----------------------------------
        app.index = 0
        app.store.set_pts(0, [[10, 10]], edited=True)
        app.handle_key(ord("2"))
        suite.equal(app.store.pts(0), [], "无预推数据时按 2 键清空生效标注")
        suite.equal(app.store.pred_pts(0), [], "无预推数据时 pred_pts 保持为空")

        # -- 边界条件 4：坐标贴边时放大镜取样不跨界、不抛异常 ----------------------------------
        for point in (
            (0, 0),
            (app.reader.width - 1, 0),
            (0, app.reader.height - 1),
            (app.reader.width - 1, app.reader.height - 1),
        ):
            app.store.set_pts(0, [list(point)], edited=True)
            app.active = 0
            app._render()
        suite.check(True, "贴边坐标的放大镜取样未抛异常")

        # -- 边界条件 5：微调在边界处被钳位 --------------------------------------------------
        app.index = 0
        app.store.set_pts(0, [[0, 0]], edited=True)
        app.active = 0
        app._nudge(-5, -5)
        suite.equal(app.store.pts(0), [[0, 0]], "贴左上角的微调被钳位")
        app.store.set_pts(0, [[app.reader.width - 1, app.reader.height - 1]], edited=True)
        app._nudge(5, 5)
        suite.equal(
            app.store.pts(0),
            [[app.reader.width - 1, app.reader.height - 1]],
            "贴右下角的微调被钳位",
        )

        # -- 边界条件 6：每次状态变更都落盘并能重新载入 ---------------------------------------
        app.store.set_pred_pts(3, [[40, 60]])
        app.store.set_pts(3, [[42, 61]], edited=True)
        app._save()
        path = app.annotation_path
        suite.check(path.is_file(), "标注文件已落盘")
        payload = json.loads(path.read_text(encoding="utf-8"))
        suite.equal(payload["video"], video.name, "落盘 JSON 的 video 字段")
        suite.equal(payload["resolution"], [app.reader.width, app.reader.height], "落盘 JSON 的 resolution")
        suite.equal(payload["frames"]["3"]["pts"], [[42, 61]], "落盘 JSON 的 pts")
        suite.equal(payload["frames"]["3"]["pred_pts"], [[40, 60]], "落盘 JSON 的 pred_pts")
        suite.equal(payload["frames"]["3"]["edited"], True, "落盘 JSON 的 edited")
        reloaded = AnnotationStore.from_dict(payload, video.name, app.reader.width, app.reader.height)
        suite.equal(reloaded.pts(3), [[42, 61]], "重载后 pts 还原")
        suite.equal(reloaded.pred_pts(3), [[40, 60]], "重载后 pred_pts 还原")
        suite.equal(reloaded.state(3), "pos", "重载后状态还原")

        # -- 边界条件 7：重新打开视频时从落盘文件恢复 -----------------------------------------
        app.quit()
        reopened = PointAnnotator(root, headless=True)
        suite.equal(reopened.store.pts(3), [[42, 61]], "重开视频后从 JSON 恢复 pts")
        suite.equal(reopened.store.pred_pts(3), [[40, 60]], "重开视频后从 JSON 恢复 pred_pts")

        # -- 边界条件 8：按 2 键可无损恢复模型预推点 -------------------------------------------
        reopened.index = 3
        reopened.active = 0
        reopened._click(200, 200, use_snap=False)
        suite.equal(reopened.store.pts(3), [[200, 200]], "误操作把点挪走了")
        reopened.handle_key(ord("2"))
        suite.equal(reopened.store.pts(3), [[40, 60]], "按 2 键无损恢复模型预推点")
        suite.equal(reopened.store.pred_pts(3), [[40, 60]], "恢复后 pred_pts 未被改动")

        # -- 边界条件 9：推进策略 ---------------------------------------------------------------
        reopened.index = 5
        reopened.store.set_pts(5, [[70, 80]], edited=True)
        reopened.store.set_pred_pts(6, [[90, 100]])
        reopened.mode = "prev"
        reopened._advance(1)
        suite.equal(reopened.store.pts(6), [[70, 80]], "Prev 模式继承上一帧的点")
        suite.equal(reopened.store.edited(6), False, "继承的帧不算人工编辑")
        suite.equal(reopened.store.pred_pts(6), [[90, 100]], "继承不改 pred_pts")

        reopened.store.set_pts(7, [[1, 1]], edited=False)
        reopened.mode = "model"
        reopened._advance(1)
        suite.equal(reopened.index, 7, "Model 模式推进到第 7 帧")
        suite.equal(reopened.store.pts(7), [[1, 1]], "Model 模式不覆盖已有标注")
        suite.equal(reopened.store.pts(8), [], "Model 模式不动没有预推点的下一帧")
        reopened.store.set_pred_pts(9, [[11, 12]])
        reopened.index = 8
        reopened._advance(1)
        suite.equal(reopened.store.pts(9), [[11, 12]], "Model 模式采用该帧的模型预推点")

        # -- 边界条件 10：三态机 ---------------------------------------------------------------
        reopened.index = 2
        reopened.handle_key(ord("n"))
        suite.equal(reopened.store.state(2), "neg", "n 键标记 neg")
        suite.equal(reopened.store.pts(2), [], "neg 帧的 pts 被清空")
        suite.equal(reopened.store.edited(2), True, "neg 帧标记为已编辑")
        reopened.index = 2
        reopened.handle_key(ord("i"))
        suite.equal(reopened.store.state(2), "ign", "i 键标记 ign")
        suite.equal(reopened.store.edited(2), True, "ign 帧标记为已编辑")

        # -- 边界条件 11：删除最后一个点回到 ign 而不是 neg -------------------------------------
        reopened.index = 4
        reopened.store.set_pts(4, [[10, 10]], edited=True)
        reopened.active = 0
        reopened.handle_key(ord("c"))
        suite.equal(reopened.store.pts(4), [], "删除最后一个点后 pts 为空")
        suite.equal(reopened.store.state(4), "ign", "删除最后一个点回到 ign")

        # -- 边界条件 12：预推点导入（逐帧 YOLO 树）---------------------------------------------
        tree_root = root / "preannot"
        sequence = tree_root / video.stem
        sequence.mkdir(parents=True)
        (sequence / "000004.txt").write_text("0 0.500000 0.250000 0.031250 0.041667\n", encoding="utf-8")
        (sequence / "scores.json").write_text(json.dumps({"000004": [0.7]}), encoding="utf-8")
        loader = PreAnnotationLoader(roots=[tree_root])
        points, source = loader.load(root, video.stem, app.reader.width, app.reader.height)
        suite.check(bool(points), "逐帧 YOLO 树被识别为预推点来源")
        suite.check(source.endswith(video.stem), "预推点来源指向正确的序列目录")
        expected_x = int(round(0.5 * app.reader.width))
        expected_y = int(round(0.25 * app.reader.height))
        suite.equal(points.get(4)["pts"], [[expected_x, expected_y]], "YOLO 框中心被换算成原图像素点")

        # -- 边界条件 13：跟踪记录 JSON 的 native → full 坐标换算 --------------------------------
        tracked = {
            "sequence": video.stem,
            "per_frame": {"7": [{"pos": [24.5, 200.0], "score": 0.5, "track_id": 1}]},
        }
        tracked_path = tree_root / f"{video.stem}.json"
        tracked_path.write_text(json.dumps(tracked), encoding="utf-8")
        native_loader = PreAnnotationLoader(roots=[tree_root], coord_space="native")
        points, _ = native_loader.load(root, video.stem, 1920, 1080)
        suite.equal(points.get(7)["pts"], [[369, 400]], "native 坐标按裁剪契约换算成原图坐标")

        # -- 边界条件 14：随机访问的帧号是精确的 -------------------------------------------------
        sequential = []
        capture = cv2.VideoCapture(str(video))
        for index in range(12):
            ok, frame = capture.read()
            if not ok:
                break
            sequential.append(frame)
        capture.release()
        reader = VideoReader(video)
        exact = all(np.array_equal(reader.frame(index), sequential[index]) for index in range(len(sequential)))
        suite.check(exact, "随机 seek 到的帧与逐帧解码像素级一致")
        reader.close()

        # -- 边界条件 15：滤镜与放大镜渲染不炸 ---------------------------------------------------
        for view in range(len(VIEW_LABELS)):
            reopened.view = view
            reopened._render()
        suite.check(True, "三种显示滤镜均能渲染")

        # -- 边界条件 16：方向键 / F 键不得误触发微调 --------------------------------------------
        suite.equal(app._decode_key(27), ("esc", ""), "ESC 归一")
        suite.equal(app._decode_key(9), ("tab", ""), "Tab 归一")
        suite.equal(app._decode_key(32), ("space", ""), "Space 归一")
        suite.equal(app._decode_key(0x01000000), ("esc", ""), "Qt 原始 ESC 归一")
        suite.equal(app._decode_key(0x01000001), ("tab", ""), "Qt 原始 Tab 归一")
        suite.equal(app._decode_key(ord("A")), ("char", "A"), "普通字符归一")
        suite.equal(app._decode_key(40 << 16)[0], "other", "Windows 方向键被挡掉")
        suite.equal(app._decode_key(0x01000013)[0], "other", "Qt 方向键被挡掉")
        # GTK 后端把键值编码成 keyval | (state << 16)：带修饰键的普通键不能被整体丢掉
        suite.equal(app._decode_key(ord("w") | (0x1 << 16)), ("char", "w"), "GTK 的 Shift+w 仍归一成字符")
        suite.equal(app._decode_key(0xFF51 | (0x1 << 16))[0], "other", "GTK 方向键被挡掉")
        suite.equal(app._decode_key(0xFF51)[0], "other", "GDK 特殊键被挡掉")
        suite.equal(app._key_modifiers(ord("w") | (0x1 << 16))[0], True, "GTK 高位掩码里的 Shift 被读出")
        suite.equal(app._key_modifiers(ord("w") | (0x8 << 16))[1], True, "GTK 高位掩码里的 Alt 被读出")
        suite.equal(app._key_modifiers(ord("w") | (0x4 << 16))[2], True, "GTK 高位掩码里的 Ctrl 被读出")
        app.index = 0
        app.store.set_pts(0, [[50, 50]], edited=True)
        app.active = 0
        app.handle_key(40 << 16)
        suite.equal(app.store.pts(0), [[50, 50]], "按方向键不会误触发微调")

        # -- 边界条件 17：鼠标坐标映射（回调坐标即画布像素，不得二次缩放）----------------------
        reopened.index = 0
        reopened.active = -1
        reopened.snap = False
        reopened.store.set_pts(0, [], edited=False)
        reopened._layout(reopened._current_frame())
        reopened.handle_mouse(cv2.EVENT_MOUSEMOVE, 400, 300)
        suite.equal(reopened.mouse_canvas, (400.0, 300.0), "鼠标回调坐标被原样当作画布像素")
        expected = reopened._to_video(400.0, 300.0)
        suite.check(
            abs(reopened.mouse_pos[0] - expected[0]) < 1e-9 and abs(reopened.mouse_pos[1] - expected[1]) < 1e-9,
            "画布坐标按 letterbox 偏移换算回原图坐标",
        )
        target = (0.5 * reopened.reader.width, 0.5 * reopened.reader.height)
        canvas_x, canvas_y = reopened._to_canvas(target[0], target[1])
        reopened.handle_mouse(cv2.EVENT_LBUTTONDOWN, canvas_x, canvas_y)
        placed = reopened.store.pts(0)
        suite.check(
            len(placed) == 1 and abs(placed[0][0] - target[0]) <= 1 and abs(placed[0][1] - target[1]) <= 1,
            f"左键点击落点与原图像素一致（got {placed}）",
        )
        reopened.mouse_pos = (-30.0, -40.0)
        reopened.handle_key(ord("p"))
        added = reopened.store.pts(0)
        suite.check(len(added) == 2 and added[1] == [0, 0], f"黑边处的光标新增点被钳到图像边界内（got {added}）")
        # 回归防护：窗口矩形变大变小都不能影响映射（旧实现按 getWindowImageRect 二次缩放，
        # 在 Qt 后端会造成约 9% 的系统性偏差）
        original_rect = cv2.getWindowImageRect
        try:
            cv2.getWindowImageRect = lambda name: (0, 0, 7, 7)
            reopened.handle_mouse(cv2.EVENT_MOUSEMOVE, 400, 300)
            suite.equal(reopened.mouse_canvas, (400.0, 300.0), "映射结果与窗口尺寸无关")
        finally:
            cv2.getWindowImageRect = original_rect

        # -- 边界条件 18：修饰键通道（Shift/Alt/Ctrl）--------------------------------------------
        mods_state = reopened.mods.state()
        suite.equal(len(mods_state), 3, "修饰键状态返回 (shift, alt, ctrl)")
        suite.check(all(isinstance(flag, bool) for flag in mods_state), "修饰键状态都是布尔值")
        if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
            # X11 路线靠 XQueryKeymap 读键盘状态；以前探针传 NULL 出参会段错误，
            # 导致整条 X11 路线被静默关掉，W/A/S/D 与 E/Q 全部失效
            suite.check(ModifierReader().available, "X11 环境下修饰键通道可用")
        original_mods = reopened.mods

        class _ShiftOn:
            available = True

            def state(self):
                return (True, False, False)

        class _NoMods:
            available = False

            def state(self):
                return (False, False, False)

        reopened.mods = _ShiftOn()
        suite.equal(reopened._shift_down(ord("w")), True, "读到 Shift 时小写 w 也按 5 px 处理")
        reopened.mods = _NoMods()
        suite.equal(reopened._shift_down(ord("W")), True, "没有修饰键通道时靠字符大小写判 Shift")
        suite.equal(reopened._shift_down(ord("w")), False, "没有修饰键通道时小写不算 Shift")
        reopened.mods = original_mods
        # X11 位图是**按 keycode 索引**的：第 0 字节覆盖 keycode 0~7，而 keycode 从 8 起
        # ⇒ 第 0 字节恒为 0，把它当「bit0=Shift」读会让整条通道永远是「无修饰键」
        x11_reader = ModifierReader()
        if x11_reader._mode == "x11":
            for name, codes in x11_reader._keycodes.items():
                code = min(codes)
                raw = bytearray(32)
                raw[code // 8] = 1 << (code % 8)
                suite.check(
                    x11_reader._x11_pressed(name, bytes(raw)),
                    f"X11 键盘位图按 keycode 取位（{name}）",
                )
            suite.check(
                not x11_reader._x11_pressed("shift", bytes(32)),
                "空的键盘位图读不出修饰键",
            )
            # 端到端：`state()` 必须走 keycode 解释。
            # 造一张「Shift 键码置位、第 0 字节为 0」的位图塞给 XQueryKeymap，
            # 若实现退化成「第 0 字节当掩码」，这里必然读出 False。
            shift_code = min(x11_reader._keycodes["shift"])
            crafted = bytearray(32)
            crafted[shift_code // 8] = 1 << (shift_code % 8)
            original_query = x11_reader._lib.XQueryKeymap

            def fake_query(display, buffer, _crafted=crafted):
                ctypes.memmove(buffer, bytes(_crafted), len(_crafted))
                return 1

            x11_reader._lib.XQueryKeymap = fake_query
            try:
                suite.equal(
                    x11_reader.state()[0],
                    True,
                    "X11 修饰键状态按 keycode 读出 Shift（第 0 字节恒为 0，不能当掩码）",
                )
            finally:
                x11_reader._lib.XQueryKeymap = original_query

        # -- 边界条件 19：按住 Alt 临时关闭吸附（用鼠标事件自带的标志位）------------------------
        captured: list[bool] = []
        original_click = reopened._click
        reopened.snap = True
        reopened._click = lambda x, y, use_snap: captured.append(use_snap)
        reopened.handle_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, cv2.EVENT_FLAG_ALTKEY)
        suite.equal(captured[-1], False, "按住 Alt 时不吸附")
        reopened.handle_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0)
        suite.equal(captured[-1], True, "没有 Alt 时按吸附开关正常吸附")
        reopened._click = original_click
        reopened.snap = False

        # -- 边界条件 20：底稿越界帧号丢弃 + state/edited 保留 + .tmp 残骸清理 ------------------
        annotations_file = root / "annotations" / f"{video.stem}.json"
        if annotations_file.is_file():
            annotations_file.unlink()
        draft_dir = root / "drafts"
        draft_dir.mkdir(parents=True)
        draft = {
            "video": video.name,
            "resolution": [app.reader.width, app.reader.height],
            "frames": {
                "2": {"state": "neg", "pts": [], "pred_pts": [[11, 12]], "edited": True},
                "3": {"state": "pos", "pts": [[21, 22]], "pred_pts": [[20, 20]], "edited": False},
                "99999": {"state": "pos", "pts": [[30, 30]], "pred_pts": [], "edited": True},
            },
        }
        (draft_dir / f"{video.stem}.json").write_text(json.dumps(draft), encoding="utf-8")
        stale_tmp = root / "annotations" / f"{video.stem}.json.stale.tmp"
        stale_tmp.write_text("{}", encoding="utf-8")
        stale_time = time.time() - 3600
        os.utime(stale_tmp, (stale_time, stale_time))
        drafted = PointAnnotator(root, preannot_dirs=[draft_dir], headless=True)
        suite.equal(drafted.store.get(99999), None, "越界帧号不会写进标注")
        suite.equal(drafted.store.state(2), "neg", "底稿的 neg 状态被保留")
        suite.equal(drafted.store.pts(2), [], "neg 帧不会因为有模型预推点而变成 pos")
        suite.equal(drafted.store.edited(2), True, "底稿的人工编辑标记被保留")
        suite.equal(drafted.store.pts(3), [[21, 22]], "底稿的生效点被导入")
        suite.equal(drafted.store.pred_pts(3), [[20, 20]], "底稿的模型预推点被导入")
        suite.check(not stale_tmp.exists(), "上次崩溃留下的 .tmp 残骸被清理")
        fresh_tmp = root / "annotations" / f"{video.stem}.json.fresh.tmp"
        fresh_tmp.write_text("{}", encoding="utf-8")
        PointAnnotator(root, headless=True)
        suite.check(fresh_tmp.exists(), "另一个实例正在写的 .tmp 不会被误删")

        # -- 边界条件 21：置信度下限过滤 --------------------------------------------------------
        score_root = root / "preannot_score"
        score_sequence = score_root / video.stem
        score_sequence.mkdir(parents=True)
        (score_sequence / "000004.txt").write_text("0 0.500000 0.250000 0.031250 0.041667\n", encoding="utf-8")
        (score_sequence / "scores.json").write_text(json.dumps({"000004": [0.7]}), encoding="utf-8")
        strict_loader = PreAnnotationLoader(roots=[score_root], min_score=0.8)
        strict_records, _ = strict_loader.load(root, video.stem, app.reader.width, app.reader.height)
        suite.equal(strict_records.get(4), None, "低于置信度下限的框被过滤掉")
        keep_loader = PreAnnotationLoader(roots=[score_root], min_score=0.5)
        keep_records, _ = keep_loader.load(root, video.stem, app.reader.width, app.reader.height)
        suite.check(bool(keep_records.get(4)), "达到置信度下限的框被保留")

        # -- 边界条件 22：播放的片尾重播与手动换帧同步 ------------------------------------------
        reopened.playing = False
        reopened.index = max(0, reopened.frame_count - 1)
        reopened._toggle_playback()
        suite.equal(reopened.index, 0, "片尾按空格从头播")
        suite.equal(reopened.playing, True, "片尾按空格开始播放")
        reopened.playing = True
        reopened._play_origin = time.monotonic()
        reopened._play_origin_index = 0
        reopened.index = 3
        reopened._advance(1)
        suite.equal(reopened._play_origin_index, reopened.index, "手动换帧后播放原点跟着走")
        reopened.playing = False

        # -- 边界条件 23：帧数未知时仍可换帧 ----------------------------------------------------
        total_frames = reopened.reader.count
        reopened.reader.count = 0
        reopened.index = 0
        reopened._advance(1)
        suite.equal(reopened.index, 1, "帧数未知时仍能前进（靠解码判断边界）")
        reopened.reader.count = total_frames

        # -- 边界条件 24：滤镜结果按帧缓存 ------------------------------------------------------
        reopened.view = 1
        reopened.index = 0
        first_filter = reopened._filtered_frame()
        second_filter = reopened._filtered_frame()
        suite.check(first_filter is second_filter, "同一帧的滤镜结果被缓存，不重复计算")
        reopened.view = 0

        # -- 边界条件 25：底稿里的 neg 帧即使自带点，导入后也必须是 neg --------------------------
        neg_dir = root / "drafts_neg"
        neg_dir.mkdir(parents=True)
        neg_draft = {
            "video": video.name,
            "resolution": [app.reader.width, app.reader.height],
            "frames": {
                "5": {"state": "neg", "pts": [[51, 52]], "pred_pts": [[51, 52]], "edited": True},
            },
        }
        (neg_dir / f"{video.stem}.json").write_text(json.dumps(neg_draft), encoding="utf-8")
        if annotations_file.is_file():
            annotations_file.unlink()
        neg_app = PointAnnotator(root, preannot_dirs=[neg_dir], headless=True)
        suite.equal(neg_app.store.state(5), "neg", "底稿的 neg 帧不会被自带点提升成 pos")
        suite.equal(neg_app.store.pts(5), [], "neg 帧导入后没有生效点")
        suite.equal(neg_app.store.pred_pts(5), [[51, 52]], "neg 帧的模型预推点仍在 pred 轨里")

        # -- 边界条件 26：帧数未知时越界帧号照样丢弃 --------------------------------------------
        unknown_dir = root / "drafts_unknown"
        unknown_dir.mkdir(parents=True)
        unknown_draft = {
            "video": video.name,
            "resolution": [app.reader.width, app.reader.height],
            "frames": {
                "1": {"state": "pos", "pts": [[10, 10]], "pred_pts": [], "edited": True},
                "99999": {"state": "pos", "pts": [[30, 30]], "pred_pts": [], "edited": True},
            },
        }
        (unknown_dir / f"{video.stem}.json").write_text(json.dumps(unknown_draft), encoding="utf-8")
        if annotations_file.is_file():
            annotations_file.unlink()
        unknown_app = PointAnnotator(root, preannot_dirs=[unknown_dir], headless=True)
        unknown_app.reader.count = 0  # 模拟「容器没写帧数」
        unknown_app._open_video(unknown_app.videos[unknown_app.video_index])
        suite.equal(unknown_app.store.get(99999), None, "帧数未知时越界帧号照样丢弃")
        suite.equal(unknown_app.store.pts(1), [[10, 10]], "帧数未知时正常帧照常导入")

        # -- 边界条件 27：帧数未知时片尾再按空格也从头播 ----------------------------------------
        unknown_app.index = 39
        unknown_app.playing = False
        unknown_app._toggle_playback()
        suite.equal(unknown_app.index, 0, "帧数未知时片尾按空格也从头播")
        unknown_app.playing = False

        # -- 边界条件 28：置信度按有效框序号索引，空行不吃分数 ----------------------------------
        messy_root = root / "preannot_messy"
        messy_sequence = messy_root / video.stem
        messy_sequence.mkdir(parents=True)
        (messy_sequence / "000004.txt").write_text(
            "\n0 0.500000 0.250000 0.031250 0.041667\n0 0.250000 0.750000 0.031250 0.041667\n",
            encoding="utf-8",
        )
        (messy_sequence / "scores.json").write_text(json.dumps({"000004": [0.1, 0.9]}), encoding="utf-8")
        messy_loader = PreAnnotationLoader(roots=[messy_root], min_score=0.5)
        messy_records, _ = messy_loader.load(root, video.stem, app.reader.width, app.reader.height)
        messy_points = messy_records.get(4, {}).get("pts", [])
        suite.equal(len(messy_points), 1, "空行不会吃掉框的置信度（只留下 0.9 分那个框）")

        # -- 边界条件 29：切换视频后滤镜缓存必须失效 --------------------------------------------
        second_video = _write_test_video(root / "VIDEO00002_test.mp4", frames=30)
        reopened.videos.append(second_video)
        reopened.video_index = len(reopened.videos) - 1
        reopened.view = 1
        reopened.index = 0
        before_switch = reopened._filtered_frame()
        reopened._open_video(second_video)
        after_switch = reopened._filtered_frame()
        suite.check(
            before_switch is not after_switch,
            "切换视频后滤镜缓存失效，不会返回上一条视频的帧",
        )

    # 走 ``log()`` 而不是裸 ``print``：PyInstaller ``--windowed`` 打包后 ``sys.stdout`` 是
    # ``None``，裸 ``print`` 会抛 ``AttributeError``，自检结果会被整个吞掉——
    # 而「在目标机上跑一次 ``--self-test``」正是 Windows 侧唯一可用的验收手段。
    log(f"[self-test] PASS {suite.passed}  FAIL {len(suite.failed)}")
    for label in suite.failed:
        log(f"[self-test] FAIL: {label}")
    return 1 if suite.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
