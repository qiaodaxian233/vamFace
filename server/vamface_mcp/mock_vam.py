"""Mock VaM —— 不开真 VaM 也能端到端跑通整套流水线的假引擎。

动机(见 对话记忆):整个项目此前卡在"真机验证"一个 gate 上。这个 mock
实现了 docs/protocol.md 的**同一套 TCP 协议**,并用 PIL 画一张由 morph 值
参数化驱动的卡通脸,于是:

  - 拟合循环 / CLI / GUI / MCP 工具,全部能在没有 VaM 的机器上端到端测试;
  - 配 --seed 生成一个"隐藏目标脸"(随机 morph 向量渲出的图),把这张图
    喂给 vamface-fit --style pixel,循环应该能把分数拉上去 —— 这是最硬的
    回归测试:优化器、桥接客户端、打分器、.vap 导出一条龙全验证;
  - 真机验证从"阻塞器"降级为"最后一步对账"。

用法:
    vamface-mock                          # 127.0.0.1:8787,零 morph 初始脸
    vamface-mock --seed 42                # 隐藏目标模式:打印目标 morph,
                                          # 并把目标渲染图存成 mock_target_42.png
    vamface-fit mock_target_42.png --style pixel --optimizer cma --iters 120

渲染器刻意做得**光滑、确定**:每个 morph 对几何是连续影响,给黑盒优化器
一个良性的地形。它画的不是 Genesis 2,而是"和 44 个精选 morph 名一一对应
的参数化卡通脸" —— 名字、协议、数值范围全按真实约定走,唯独渲染是假的。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import random
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

from .morph_presets import FACE_MORPH_GROUPS

log = logging.getLogger("vamface.mock")

MOCK_VERSION = "mock-0.3"

# morph 名 → (分组, min, max)。全量支持 44 个精选 morph,范围沿 VaM 惯例。
MORPH_DEFS: Dict[str, Tuple[str, float, float]] = {
    name: (group, -1.5, 1.5)
    for group, names in FACE_MORPH_GROUPS.items()
    for name in names
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


# ---------------------------------------------------------------------------
# 参数化人脸渲染器
# ---------------------------------------------------------------------------

class FaceRenderer:
    """morph dict → 512x512 RGB 图。所有几何量都是 morph 的连续函数。"""

    SIZE = 512

    def __init__(self, skin_rgb: Tuple[int, int, int] = (236, 200, 172)) -> None:
        self.skin_rgb = skin_rgb

    def render(self, m: Dict[str, float]):
        from PIL import Image, ImageDraw

        g = lambda name: float(m.get(name, 0.0))  # noqa: E731
        S = self.SIZE
        img = Image.new("RGB", (S, S), (245, 245, 248))
        d = ImageDraw.Draw(img)
        cx, cy = S / 2, S * 0.52

        # ---- 头部轮廓:上半椭圆 + 下颌曲线 ----------------------------------
        head = 1.0 + 0.18 * g("Head Big") + 0.15 * g("Head Scale")
        fw = 150 * head * (1 + 0.15 * g("Face Round") - 0.08 * g("Face Long")
                           + 0.05 * g("Face Flat"))
        fh = 195 * head * (1 + 0.15 * g("Face Long") - 0.08 * g("Face Round")
                           + 0.05 * g("Cranium Shape"))
        jaw_w = fw * (0.72 + 0.12 * g("Jaw Width") + 0.06 * g("Cheekbones Width"))
        chin_w = fw * (0.26 + 0.10 * g("Chin Width"))
        chin_y = cy + fh * (0.98 + 0.10 * g("Chin Height") + 0.08 * g("Jaw Size")
                            + 0.05 * g("Jaw Height"))
        chin_fwd = 6 * g("Chin Forward") + 4 * g("Chin Depth")  # 平移近似"前伸"
        jaw_y = cy + fh * (0.55 + 0.06 * g("Jaw Angle"))

        pts: List[Tuple[float, float]] = []
        import math
        for i in range(37):  # 上半 + 颧骨段的椭圆采样(180°..360°)
            a = math.pi + math.pi * i / 36
            pts.append((cx + fw * math.cos(a), cy + fh * 0.92 * math.sin(a)))
        # 右颊 → 右颌 → 下巴 → 左颌 → 左颊
        cheek = 1 + 0.08 * g("Cheekbones Size") - 0.06 * g("Cheeks Sink") \
            + 0.04 * g("Cheeks Depth")
        pts += [(cx + fw * cheek, cy + fh * 0.15),
                (cx + jaw_w, jaw_y),
                (cx + chin_w + chin_fwd, chin_y),
                (cx - chin_w + chin_fwd, chin_y),
                (cx - jaw_w, jaw_y),
                (cx - fw * cheek, cy + fh * 0.15)]
        d.polygon(pts, fill=self.skin_rgb, outline=(120, 90, 75))

        # ---- 耳朵 ----------------------------------------------------------
        ear_r = 22 * (1 + 0.35 * g("Ears Size"))
        ear_y = cy - fh * 0.05 - 25 * g("Ears Height")
        for sx in (-1, 1):
            ex = cx + sx * fw * 0.98
            d.ellipse([ex - ear_r, ear_y - ear_r * 1.4,
                       ex + ear_r, ear_y + ear_r * 1.4],
                      fill=self.skin_rgb, outline=(120, 90, 75))

        # ---- 眼睛 ----------------------------------------------------------
        eye_gap = fw * (0.46 + 0.14 * g("Eyes Width Spacing"))
        eye_y = cy - fh * (0.18 - 0.06 * g("Eyes Height"))
        eye_w = 30 * (1 + 0.35 * g("Eyes Size"))
        eye_h = 18 * (1 + 0.35 * g("Eyes Size") - 0.30 * g("Eyelids Height")
                      + 0.10 * g("Eye Fold Depth"))
        slant = 10 * g("Eyes Slant")
        depth_shade = int(20 * max(0.0, g("Eyes Depth")))
        for sx in (-1, 1):
            ex = cx + sx * eye_gap / 2
            ey = eye_y - sx * slant  # 左右反向偏移 → 吊梢/下垂
            if depth_shade:  # 眼窝阴影近似"深度"
                d.ellipse([ex - eye_w * 1.2, ey - eye_h * 1.5,
                           ex + eye_w * 1.2, ey + eye_h * 1.5],
                          fill=(max(0, self.skin_rgb[0] - depth_shade),
                                max(0, self.skin_rgb[1] - depth_shade),
                                max(0, self.skin_rgb[2] - depth_shade)))
            d.ellipse([ex - eye_w, ey - eye_h, ex + eye_w, ey + eye_h],
                      fill=(250, 250, 250), outline=(60, 50, 45), width=2)
            r = eye_h * 0.75
            d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(70, 60, 120))
            # 眉毛
            by = ey - eye_h - 16 - 10 * g("Brow Height")
            d.line([ex - eye_w, by + 3 * sx * 0, ex + eye_w, by - slant * 0.4],
                   fill=(90, 70, 60), width=5)

        # ---- 鼻子 ----------------------------------------------------------
        nose_len = fh * (0.20 + 0.08 * g("Nose Size") + 0.06 * g("Nose Height"))
        nose_w = 16 * (1 + 0.30 * g("Nose Width") + 0.15 * g("Nostrils Width"))
        bridge_w = 6 * (1 + 0.4 * g("Nose Bridge Width"))
        tip_y = eye_y + nose_len - 6 * g("Nose Tip Height")
        tip_w = nose_w * (0.6 + 0.2 * g("Nose Tip Width"))
        bump = 4 * g("Nose Bump")
        d.polygon([(cx - bridge_w, eye_y + 8), (cx + bridge_w, eye_y + 8),
                   (cx + tip_w + bump, tip_y), (cx - tip_w - bump, tip_y)],
                  outline=(120, 90, 75))
        for sx in (-1, 1):  # 鼻孔
            d.ellipse([cx + sx * nose_w * 0.7 - 3, tip_y - 3,
                       cx + sx * nose_w * 0.7 + 3, tip_y + 3], fill=(120, 90, 75))

        # ---- 嘴 ------------------------------------------------------------
        mouth_y = tip_y + fh * (0.16 - 0.05 * g("Mouth Height"))
        mouth_w = 44 * (1 + 0.25 * g("Mouth Size") + 0.20 * g("Mouth Width")
                        + 0.10 * g("Lips Width"))
        lip_h = 10 * (1 + 0.30 * g("Lips Thickness")
                      + 0.15 * g("Upper Lip Thickness")
                      + 0.15 * g("Lower Lip Thickness")
                      + 0.10 * g("Mouth Size"))
        corner = 8 * g("Mouth Corners")
        d.ellipse([cx - mouth_w, mouth_y - lip_h, cx + mouth_w, mouth_y + lip_h],
                  fill=(196, 106, 110), outline=(120, 60, 60))
        d.arc([cx - mouth_w, mouth_y - abs(corner) - 2,
               cx + mouth_w, mouth_y + abs(corner) + 2],
              start=180 if corner >= 0 else 0,
              end=360 if corner >= 0 else 180, fill=(120, 60, 60), width=2)
        return img

    def render_array(self, m: Dict[str, float]):
        import numpy as np
        return np.asarray(self.render(m))


# ---------------------------------------------------------------------------
# 协议服务器
# ---------------------------------------------------------------------------

class MockVamServer:
    """实现 docs/protocol.md 全部命令的假 VaM。线程化,支持多连接。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8787) -> None:
        self.host = host
        self.port = port
        self._srv: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # 场景状态
        self.morphs: Dict[str, float] = {n: 0.0 for n in MORPH_DEFS}
        self.characters = ["MockPale", "MockTan", "MockOlive"]
        self.character = self.characters[0]
        self._skins = {"MockPale": (236, 200, 172), "MockTan": (198, 150, 116),
                       "MockOlive": (176, 142, 104)}
        # 通用参数区(皮肤 L0 / 连接调试页要用)
        self.params: Dict[str, Dict[str, Dict[str, Any]]] = {
            "skin": {
                "Skin Color": {"type": "color", "value": {"h": 0.08, "s": 0.27, "v": 0.93}},
                "Gloss": {"type": "float", "value": 0.5, "min": 0.0, "max": 1.0},
                "Use Advanced Colors": {"type": "bool", "value": False},
            },
            "textures": {
                "faceDiffuseUrl": {"type": "string", "value": ""},
                "torsoDiffuseUrl": {"type": "string", "value": ""},
            },
            "geometry": {
                "character": {"type": "chooser", "value": self.character,
                              "choices": list(self.characters)},
            },
        }
        self.actions = {"skin": ["Reset Skin"], "geometry": ["Reload"]}
        self.renderer = FaceRenderer(self._skins[self.character])

    # -- 生命周期 -------------------------------------------------------------

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        self.port = srv.getsockname()[1]  # port=0 时拿实际端口(测试用)
        srv.listen(8)
        srv.settimeout(0.5)
        self._srv = srv
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        self._threads.append(t)
        log.info("MockVaM listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._client_loop, args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    def _client_loop(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        try:
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    break
                try:
                    req = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    resp = {"id": "", "ok": False, "error": "bad json"}
                else:
                    resp = self._dispatch(req)
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            try:
                f.close()
                conn.close()
            except OSError:
                pass

    # -- 命令分发 -------------------------------------------------------------

    def _dispatch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        rid = str(req.get("id", ""))
        cmd = req.get("cmd", "")
        args = req.get("args") or {}
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return {"id": rid, "ok": False, "error": f"unknown cmd: {cmd}"}
        try:
            with self._lock:
                data = handler(args)
            return {"id": rid, "ok": True, "data": data}
        except CommandError as e:
            return {"id": rid, "ok": False, "error": str(e)}
        except Exception as e:  # mock 内部 bug 也按协议报错,不让连接死掉
            log.exception("mock handler crashed")
            return {"id": rid, "ok": False, "error": f"mock internal: {e}"}

    def _need_atom(self, args: Dict[str, Any]) -> None:
        atom = args.get("atom", "Person")
        if atom != "Person":
            raise CommandError(f"atom not found: {atom}")

    # -- 命令实现(与 protocol.md 表一一对应)----------------------------------

    def _cmd_ping(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"version": MOCK_VERSION, "app": "MockVaM"}

    def _cmd_list_atoms(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"atoms": [{"uid": "Person", "type": "Person"},
                          {"uid": "MockLight", "type": "InvisibleLight"}]}

    def _cmd_list_morphs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        filt = str(args.get("filter", "")).lower()
        region = str(args.get("region", "")).lower()
        limit = int(args.get("limit", 200))
        rows = []
        for name, (grp, lo, hi) in MORPH_DEFS.items():
            if filt and filt not in name.lower():
                continue
            if region and region != grp:
                continue
            rows.append({"name": name, "uid": f"mock/{name}", "region": grp,
                         "value": self.morphs[name], "min": lo, "max": hi})
        return {"count": len(rows[:limit]), "total": len(rows),
                "morphs": rows[:limit]}

    def _cmd_get_morphs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        changed_only = bool(args.get("changed_only", True))
        vals = {n: v for n, v in self.morphs.items()
                if not changed_only or abs(v) > 1e-9}
        return {"values": vals}

    def _cmd_set_morphs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        values = args.get("values") or {}
        clamp = bool(args.get("clamp", True))
        applied, missing = 0, []
        for name, v in values.items():
            if name not in MORPH_DEFS:
                missing.append(name)  # 教训5:没有的 morph 报 missing 不报错
                continue
            _, lo, hi = MORPH_DEFS[name]
            self.morphs[name] = _clamp(v, lo, hi) if clamp else float(v)
            applied += 1
        return {"applied": applied, "missing": missing}

    def _cmd_reset_morphs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        n = sum(1 for v in self.morphs.values() if abs(v) > 1e-9)
        self.morphs = {k: 0.0 for k in self.morphs}
        return {"reset": n}

    def _cmd_load_scene(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not args.get("path"):
            raise CommandError("load_scene requires 'path'")
        return {"loading": True}

    def _cmd_focus_head(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        return {"focused": "Person/headControl"}

    def _cmd_screenshot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        max_width = int(args.get("max_width", 0) or 0)
        img = self.renderer.render(self.morphs)
        if max_width and img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"width": img.width, "height": img.height,
                "png_base64": base64.b64encode(buf.getvalue()).decode("ascii")}

    def _cmd_list_storables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        return {"storables": sorted(self.params)}

    def _params_of(self, args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        self._need_atom(args)
        sid = args.get("storable", "")
        if sid not in self.params:
            raise CommandError(f"storable not found: {sid}")
        return self.params[sid]

    def _cmd_list_params(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ps = self._params_of(args)
        out: Dict[str, List[str]] = {"floats": [], "bools": [], "colors": [],
                                     "choosers": [], "strings": []}
        for name, p in ps.items():
            out[{"float": "floats", "bool": "bools", "color": "colors",
                 "chooser": "choosers", "string": "strings"}[p["type"]]].append(name)
        return out

    def _cmd_get_param(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ps = self._params_of(args)
        name = args.get("param", "")
        if name not in ps:
            raise CommandError(f"param not found: {name}")
        return dict(ps[name])

    def _cmd_set_param(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ps = self._params_of(args)
        name = args.get("param", "")
        if name not in ps:
            raise CommandError(f"param not found: {name}")
        p = ps[name]
        p["value"] = args.get("value")
        # 假引擎的"真实效果":写 Skin Color 会改渲染肤色(HSV 假定 0-1,
        # 与 skin.rgb_to_vam_hsv 的假设一致 —— 真机验证若推翻,两边一起改)
        if name == "Skin Color" and isinstance(p["value"], dict):
            import colorsys
            h = float(p["value"].get("h", 0.0))
            s = float(p["value"].get("s", 0.0))
            v = float(p["value"].get("v", 0.0))
            rgb = tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h, s, v))
            self.renderer.skin_rgb = rgb  # type: ignore[assignment]
        return {"type": p["type"], "value": p["value"]}

    def _cmd_call_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        sid, action = args.get("storable", ""), args.get("action", "")
        if action not in self.actions.get(sid, []):
            raise CommandError(f"action not found: {sid}/{action}")
        return {"called": action}

    def _cmd_list_characters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        return {"characters": list(self.characters)}

    def _cmd_set_character(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._need_atom(args)
        name = args.get("name", "")
        if name not in self.characters:
            raise CommandError(f"character not found: {name}")
        self.character = name
        self.renderer.skin_rgb = self._skins[name]
        self.params["geometry"]["character"]["value"] = name
        return {"selected": name}


class CommandError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 隐藏目标模式 + CLI
# ---------------------------------------------------------------------------

def make_hidden_target(seed: int, n_morphs: int = 12,
                       amplitude: float = 0.8) -> Dict[str, float]:
    """随机挑 n 个 morph 给非零值,作为拟合测试的"标准答案"。"""
    rng = random.Random(seed)
    names = rng.sample(sorted(MORPH_DEFS), k=min(n_morphs, len(MORPH_DEFS)))
    return {n: round(rng.uniform(-amplitude, amplitude), 3) for n in names}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="vamface-mock",
        description="假 VaM:同协议 TCP server + 参数化卡通脸渲染,"
                    "无需真 VaM 即可端到端测试整条流水线。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--seed", type=int,
                        help="隐藏目标模式:用该种子随机生成目标 morph,"
                             "渲染目标图存盘并打印标准答案")
    parser.add_argument("--n-morphs", type=int, default=12,
                        help="隐藏目标里非零 morph 的个数(默认 12)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    srv = MockVamServer(args.host, args.port)

    if args.seed is not None:
        target = make_hidden_target(args.seed, args.n_morphs)
        img = FaceRenderer().render(target)
        out = f"mock_target_{args.seed}.png"
        img.save(out)
        print(f"隐藏目标已生成: {out}")
        print("标准答案(拟合应逼近这些值):")
        print(json.dumps(target, ensure_ascii=False, indent=2))
        print(f"\n下一步:  vamface-fit {out} --style pixel --port {args.port}")

    srv.start()
    print(f"MockVaM 运行中 @ {srv.host}:{srv.port} (Ctrl+C 退出)")
    try:
        _serve_wait()
    except KeyboardInterrupt:
        srv.stop()
    return 0


def _serve_wait() -> None:  # 单独一层:测试里 monkeypatch 这个即可立即返回
    threading.Event().wait()


if __name__ == "__main__":
    import sys
    sys.exit(main())
