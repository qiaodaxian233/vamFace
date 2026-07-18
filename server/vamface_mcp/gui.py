"""vamFace GUI — 本地网页操作台 (Gradio).

不依赖 MCP:直接通过 TCP 连 VaM 里的 VamFaceBridge 插件。

    pip install -e ".[gui]"      # gradio + matplotlib
    vamface-gui                  # 打开 http://127.0.0.1:7860

三个标签页:
  照片拟合  — 拖照片、选优化器、实时看"目标 vs 当前渲染"和分数曲线、导出 .vap
  皮肤 L0   — 采样照片肤色、切换角色皮肤、把肤色写进颜色参数
  连接调试  — ping / atoms / storables / 参数查看

设计约定(沿用 对话记忆 的教训):
  - 所有桥接调用失败都以文字显示在界面上,不弹异常、不阻塞;
  - 参数名靠运行时扫描发现,不硬编码猜测。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .bridge_client import BridgeClient, BridgeError
from .fitting import FitConfig, fit_face
from .morph_presets import FACE_MORPH_GROUPS, default_bounds
from .skin import apply_skin_color, sample_skin_tone
from .vap import write_vap

OUT_DIR = Path("./out").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

_bridge: Optional[BridgeClient] = None


def _get_bridge(host: str, port: float) -> BridgeClient:
    global _bridge
    port = int(port)
    if _bridge is None or _bridge.host != host or _bridge.port != port:
        if _bridge is not None:
            _bridge.close()
        _bridge = BridgeClient(host, port)
    return _bridge


def _swatch(rgb: List[int], size: int = 96) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = np.array(rgb, dtype=np.uint8)
    return img


def _score_plot(history: List[float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    if history:
        best = np.maximum.accumulate(np.asarray(history, dtype=float))
        ax.plot(history, lw=0.8, alpha=0.5, label="每次评估")
        ax.plot(best, lw=1.8, label="历史最优")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("identity score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1: 照片拟合
# ---------------------------------------------------------------------------

def run_fit(host, port, photo_path, atom, optimizer, iters, groups, width):
    """Generator: streams live progress to the UI while fitting runs."""
    if not photo_path:
        yield None, None, "请先上传目标照片", None, None
        return

    bridge = _get_bridge(host, port)
    try:
        info = bridge.ping()
    except BridgeError as e:
        yield None, None, f"❌ 连不上桥接插件: {e}", None, None
        return

    names = [n for g in (groups or FACE_MORPH_GROUPS.keys())
             for n in FACE_MORPH_GROUPS.get(g, [])]

    state: Dict[str, Any] = {"history": [], "last_img": None,
                             "done": False, "error": None, "result": None}

    def on_eval(count: int, score: float, img: np.ndarray) -> None:
        state["history"].append(score)
        state["last_img"] = img

    cfg = FitConfig(atom=atom, morph_names=names,
                    bounds={n: default_bounds(n) for n in names},
                    max_iters=int(iters), screenshot_width=int(width),
                    on_eval=on_eval)

    def worker() -> None:
        try:
            state["result"] = fit_face(bridge, photo_path, cfg, optimizer=optimizer)
        except Exception as e:  # BridgeError or optimizer failure
            state["error"] = str(e)
        finally:
            state["done"] = True

    threading.Thread(target=worker, daemon=True).start()

    from PIL import Image
    target_img = np.asarray(Image.open(photo_path).convert("RGB"))

    while not state["done"]:
        n = len(state["history"])
        best = max(state["history"]) if state["history"] else 0.0
        status = (f"运行中 · 已连 VamFaceBridge v{info.get('version')} · "
                  f"评估 {n}/{int(iters)} · 当前最优 {best:.4f}")
        yield target_img, state["last_img"], status, _score_plot(state["history"]), None
        time.sleep(0.8)

    if state["error"]:
        yield target_img, state["last_img"], f"❌ 拟合失败: {state['error']}", \
            _score_plot(state["history"]), None
        return

    result = state["result"]
    vap_path = OUT_DIR / f"fit_{int(time.time())}.vap"
    write_vap(vap_path, result.best_morphs)
    status = f"✅ 完成 · 最优分 {result.best_score:.4f} · 已存 {vap_path.name}"
    if result.warning:
        status += f"\n⚠️ {result.warning} — 分数是占位值,装拟合依赖: pip install -e '.[fit]'"
    yield target_img, state["last_img"], status, _score_plot(state["history"]), str(vap_path)


# ---------------------------------------------------------------------------
# Tab 2: 皮肤 Level 0
# ---------------------------------------------------------------------------

def do_sample_tone(photo_path):
    if not photo_path:
        return None, "请先上传照片"
    try:
        tone = sample_skin_tone(photo_path)
        info = (f"肤色 {tone['hex']} · RGB {tone['rgb']} · "
                f"检测: {tone['detector']} / {tone['mask']} · "
                f"采样像素 {tone['pixels_used']}")
        return _swatch(tone["rgb"]), info
    except Exception as e:
        return None, f"❌ 采样失败: {e}"


def do_list_characters(host, port, atom):
    try:
        chars = _get_bridge(host, port).list_characters(atom)
        import gradio as gr
        return gr.update(choices=chars,
                         value=chars[0] if chars else None), f"找到 {len(chars)} 个皮肤/角色"
    except BridgeError as e:
        import gradio as gr
        return gr.update(), f"❌ {e}"


def do_set_character(host, port, atom, name):
    if not name:
        return "请先扫描并选择一个角色"
    try:
        _get_bridge(host, port).set_character(atom, name)
        return f"✅ 已切换到: {name}"
    except BridgeError as e:
        return f"❌ {e}"


def do_scan_color_params(host, port, atom, storable_filter):
    try:
        hits = _get_bridge(host, port).find_color_params(atom, storable_filter or "")
        choices = [f"{h['storable']} :: {h['param']}" for h in hits]
        import gradio as gr
        return gr.update(choices=choices, value=choices[:1]), \
            f"扫到 {len(choices)} 个颜色参数(过滤词: '{storable_filter}')"
    except BridgeError as e:
        import gradio as gr
        return gr.update(), f"❌ {e}"


def do_apply_tone(host, port, atom, photo_path, selected_params):
    if not photo_path:
        return "请先上传照片"
    if not selected_params:
        return "请先扫描并勾选要写入的颜色参数"
    try:
        tone = sample_skin_tone(photo_path)
    except Exception as e:
        return f"❌ 采样失败: {e}"
    bridge = _get_bridge(host, port)
    lines = [f"肤色 {tone['hex']} →"]
    for item in selected_params:
        storable, _, param = item.partition(" :: ")
        try:
            apply_skin_color(bridge, atom, storable.strip(), param.strip(), tone["rgb"])
            lines.append(f"  ✅ {item}")
        except Exception as e:
            lines.append(f"  ❌ {item}: {e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3: 连接调试
# ---------------------------------------------------------------------------

def do_ping(host, port):
    try:
        info = _get_bridge(host, port).ping()
        return f"✅ VamFaceBridge v{info.get('version')} @ {host}:{int(port)}"
    except BridgeError as e:
        return f"❌ {e}"


def do_list_atoms(host, port):
    try:
        data = _get_bridge(host, port).list_atoms()
        atoms = data.get("atoms") or []
        return "\n".join(f"{a['type']:<16} {a['uid']}" for a in atoms) or "(空场景)"
    except BridgeError as e:
        return f"❌ {e}"


def do_inspect(host, port, atom, storable):
    bridge = _get_bridge(host, port)
    try:
        if not storable:
            return json.dumps(bridge.list_storables(atom), ensure_ascii=False, indent=2)
        return json.dumps(bridge.list_params(atom, storable), ensure_ascii=False, indent=2)
    except BridgeError as e:
        return f"❌ {e}"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr

    with gr.Blocks(title="vamFace") as demo:
        gr.Markdown("# vamFace · 照片 → VaM 面部\n"
                    "需要 VaM 运行中且已加载 VamFaceBridge session 插件。")
        with gr.Row():
            host = gr.Textbox(value="127.0.0.1", label="Host", scale=2)
            port = gr.Number(value=8787, label="Port", precision=0, scale=1)
            ping_btn = gr.Button("测试连接", scale=1)
            ping_out = gr.Markdown("")
        ping_btn.click(do_ping, [host, port], ping_out)

        with gr.Tab("照片拟合"):
            with gr.Row():
                photo = gr.Image(label="目标照片", type="filepath", height=320)
                current = gr.Image(label="VaM 当前渲染", height=320)
            with gr.Row():
                atom = gr.Textbox(value="Person", label="Person 原子", scale=1)
                optimizer = gr.Radio(["cma", "greedy"], value="cma", label="优化器", scale=1)
                iters = gr.Slider(10, 400, value=60, step=10, label="评估预算", scale=2)
                width = gr.Slider(256, 1024, value=512, step=64, label="截图宽度", scale=1)
            groups = gr.CheckboxGroup(sorted(FACE_MORPH_GROUPS),
                                      value=sorted(FACE_MORPH_GROUPS),
                                      label="morph 分组(留空=全部;建议先 skull+jaw 粗调)")
            fit_btn = gr.Button("开始拟合", variant="primary")
            fit_status = gr.Markdown("")
            fit_plot = gr.Plot(label="分数曲线")
            fit_vap = gr.File(label="导出的 .vap 预设")
            fit_btn.click(run_fit, [host, port, photo, atom, optimizer, iters, groups, width],
                          [photo, current, fit_status, fit_plot, fit_vap])

        with gr.Tab("皮肤 L0"):
            gr.Markdown("Level 0 = 采样照片肤色 → 选最接近的皮肤 → 微调颜色参数。"
                        "参数名靠现场扫描,不硬编码。")
            skin_photo = gr.Image(label="照片", type="filepath", height=280)
            with gr.Row():
                sample_btn = gr.Button("采样肤色")
                swatch = gr.Image(label="肤色", height=96, width=96)
                tone_info = gr.Markdown("")
            sample_btn.click(do_sample_tone, [skin_photo], [swatch, tone_info])

            skin_atom = gr.Textbox(value="Person", label="Person 原子")
            with gr.Row():
                chars_btn = gr.Button("扫描皮肤/角色")
                chars_dd = gr.Dropdown(choices=[], label="角色/皮肤")
                apply_char_btn = gr.Button("应用该皮肤")
            chars_status = gr.Markdown("")
            chars_btn.click(do_list_characters, [host, port, skin_atom],
                            [chars_dd, chars_status])
            apply_char_btn.click(do_set_character, [host, port, skin_atom, chars_dd],
                                 chars_status)

            with gr.Row():
                filt = gr.Textbox(value="skin", label="storable 过滤词")
                scan_btn = gr.Button("扫描颜色参数")
            color_params = gr.CheckboxGroup(choices=[], label="要写入的颜色参数")
            scan_status = gr.Markdown("")
            scan_btn.click(do_scan_color_params, [host, port, skin_atom, filt],
                           [color_params, scan_status])
            apply_tone_btn = gr.Button("把采样肤色写入所选参数", variant="primary")
            apply_status = gr.Markdown("")
            apply_tone_btn.click(do_apply_tone,
                                 [host, port, skin_atom, skin_photo, color_params],
                                 apply_status)

        with gr.Tab("连接调试"):
            atoms_btn = gr.Button("列出场景原子")
            atoms_out = gr.Code(label="atoms")
            atoms_btn.click(do_list_atoms, [host, port], atoms_out)
            with gr.Row():
                dbg_atom = gr.Textbox(value="Person", label="原子")
                dbg_storable = gr.Textbox(value="", label="storable(留空=列出全部 id)")
                dbg_btn = gr.Button("查看")
            dbg_out = gr.Code(label="结果", language="json")
            dbg_btn.click(do_inspect, [host, port, dbg_atom, dbg_storable], dbg_out)

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()
