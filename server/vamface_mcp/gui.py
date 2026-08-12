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

# --- direct-run bootstrap -------------------------------------------------
# 包内模块被当成裸文件跑(Windows 上双击 / 直接敲文件路径)时,相对导入会炸
# "attempted relative import with no known parent package"。检测到这种情况就
# 把包父目录塞进 sys.path,以正确的模块身份重跑一遍。用户实测踩过(2026-08-12)。
if __package__ in (None, ""):
    import os as _os, sys as _sys, runpy as _runpy
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    _runpy.run_module("vamface_mcp.gui", run_name="__main__")
    _sys.exit(0)
# ---------------------------------------------------------------------------


import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__
from .bridge_client import BridgeClient, BridgeError
from .fitting import FitConfig, decode_png_b64, fit_face
from .morph_presets import FACE_MORPH_GROUPS, default_bounds
from .skin import apply_skin_color, sample_skin_tone
from .updater import (check_latest, install_plugin, load_config,
                      plugin_source_version, save_config)
from .vap import read_vap, write_vap

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

def run_fit(host, port, photo_path, atom, optimizer, style, c2f, iters, groups, width):
    """Generator: streams live progress to the UI while fitting runs."""
    if not photo_path:
        yield None, None, "请先上传目标照片", None, None
        return

    bridge = _get_bridge(host, port)
    try:
        info = bridge.handshake()
    except BridgeError as e:
        yield None, None, f"❌ 连不上桥接插件: {e}", None, None
        return
    compat = f"\n⚠️ {info['compat_warning']}" if info.get("compat_warning") else ""

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
            state["result"] = fit_face(bridge, photo_path, cfg,
                                       optimizer=optimizer, style=style,
                                       coarse_to_fine=bool(c2f))
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
    status = (f"✅ 完成 · 最优分 {result.best_score:.4f} · "
              f"打分 {result.style}/{result.scorer_name} · "
              f"缓存命中 {result.cache_hits} · 已存 {vap_path.name}") + compat
    if result.hints:
        status += "\n" + "\n".join(f"💡 {h}" for h in result.hints[:5])
    if result.warning:
        status += f"\n⚠️ {result.warning}"
        if "NullScorer" in result.warning:
            status += "(分数是占位值,装拟合依赖: pip install -e '.[fit]')"
    if result.missing:
        shown = ", ".join(result.missing[:6])
        more = f" 等 {len(result.missing)} 个" if len(result.missing) > 6 else ""
        status += (f"\n⚠️ 目标 VaM 缺精选 morph: {shown}{more}"
                   f" — 这些维度没参与拟合,把完整列表发给开发者校准 morph_presets")
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
        info = _get_bridge(host, port).handshake()
        msg = f"✅ VamFaceBridge v{info.get('version')} @ {host}:{int(port)}"
        if info.get("compat_warning"):
            msg += f"\n⚠️ {info['compat_warning']}"
        return msg
    except BridgeError as e:
        return f"❌ {e}"


def do_check_update():
    r = check_latest()
    if r["error"]:
        return f"server v{r['installed']} · ⚠️ {r['error']}"
    if r["update_available"]:
        return (f"server v{r['installed']} → GitHub 最新 v{r['latest']} · "
                f"更新: 仓库目录里 `git pull` 后重启;插件用下方连接调试页一键更新")
    return f"server v{r['installed']} · ✅ 已是最新(GitHub: v{r['latest']})"


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
# Tab: 手动微调(v0.5)
# ---------------------------------------------------------------------------

TUNE_NAMES: List[str] = [n for g in sorted(FACE_MORPH_GROUPS)
                         for n in FACE_MORPH_GROUPS[g]]


def tune_load(host, port, atom):
    """读取 VaM 当前值 → 刷新全部滑块。缺的 morph 显示 0 并在状态里点名。"""
    import gradio as gr
    try:
        vals = _get_bridge(host, port).get_morphs(atom, changed_only=False)
    except BridgeError as e:
        return [gr.update() for _ in TUNE_NAMES] + [f"❌ {e}"]
    missing = [n for n in TUNE_NAMES if n not in vals]
    status = f"✅ 已读取 {len(TUNE_NAMES) - len(missing)}/{len(TUNE_NAMES)} 个 morph"
    if missing:
        status += f" · 目标 VaM 没有: {', '.join(missing[:8])}" + \
                  (" …" if len(missing) > 8 else "")
    return [gr.update(value=float(vals.get(n, 0.0))) for n in TUNE_NAMES] + [status]


def tune_set_one(name, host, port, atom, value):
    try:
        r = _get_bridge(host, port).set_morphs(atom, {name: float(value)})
        if r.get("missing"):
            return f"⚠️ 目标 VaM 没有这个 morph: {name}"
        return f"✅ {name} = {float(value):.3f}"
    except BridgeError as e:
        return f"❌ {e}"


def tune_zero(host, port, atom):
    import gradio as gr
    try:
        _get_bridge(host, port).set_morphs(atom, {n: 0.0 for n in TUNE_NAMES})
    except BridgeError as e:
        return [gr.update() for _ in TUNE_NAMES] + [f"❌ {e}"]
    return [gr.update(value=0.0) for n in TUNE_NAMES] + ["✅ 精选 morph 已全部归零"]


def tune_shot(host, port, atom):
    try:
        b = _get_bridge(host, port)
        try:
            b.focus_head(atom)
        except BridgeError:
            pass  # 聚焦失败不挡截图
        shot = b.screenshot(max_width=640)
        return decode_png_b64(shot["png_base64"]), "✅ 已刷新"
    except BridgeError as e:
        return None, f"❌ {e}"


def tune_export(*slider_values):
    vals = {n: float(v) for n, v in zip(TUNE_NAMES, slider_values)
            if abs(float(v)) > 1e-6}
    path = OUT_DIR / f"tune_{int(time.time())}.vap"
    write_vap(path, vals)
    return str(path), f"✅ 已导出 {path.name}({len(vals)} 个非零 morph)"


def tune_apply_vap(host, port, atom, vap_file):
    import gradio as gr
    if not vap_file:
        return [gr.update() for _ in TUNE_NAMES] + ["请先选择 .vap 文件"]
    try:
        morphs = read_vap(vap_file)
        r = _get_bridge(host, port).set_morphs(atom, morphs)
    except (BridgeError, OSError, ValueError, KeyError) as e:
        return [gr.update() for _ in TUNE_NAMES] + [f"❌ {e}"]
    status = f"✅ 已应用 {r.get('applied', 0)} 个 morph"
    if r.get("missing"):
        status += f" · 缺失 {len(r['missing'])} 个"
    return [gr.update(value=float(morphs.get(n, 0.0)))
            for n in TUNE_NAMES] + [status]


# ---------------------------------------------------------------------------
# 插件更新(v0.5)
# ---------------------------------------------------------------------------

def do_install_plugin(vam_dir):
    if not vam_dir or not vam_dir.strip():
        return "请填 VaM 安装目录(含 VaM.exe 的那个)"
    r = install_plugin(vam_dir.strip())
    if not r["ok"]:
        return f"❌ {r['error']}"
    save_config(vam_dir=vam_dir.strip())
    lines = [f"✅ 插件已写入 {r['dest']}(v{plugin_source_version()})"]
    if r["backup"]:
        lines.append(f"旧文件备份: {r['backup']}")
    if r["note"]:
        lines.append(r["note"])
    lines.append("最后一步得人做:VaM → Session Plugins → 该插件点 Reload。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr

    with gr.Blocks(title="vamFace") as demo:
        gr.Markdown(f"# vamFace v{__version__} · 照片 → VaM 面部\n"
                    "需要 VaM 运行中且已加载 VamFaceBridge session 插件。")
        with gr.Row():
            host = gr.Textbox(value="127.0.0.1", label="Host", scale=2)
            port = gr.Number(value=8787, label="Port", precision=0, scale=1)
            ping_btn = gr.Button("测试连接", scale=1)
            upd_btn = gr.Button("检查更新", scale=1)
            ping_out = gr.Markdown("")
        ping_btn.click(do_ping, [host, port], ping_out)
        upd_btn.click(do_check_update, [], ping_out)

        with gr.Tab("照片拟合"):
            with gr.Row():
                photo = gr.Image(label="目标照片", type="filepath", height=320)
                current = gr.Image(label="VaM 当前渲染", height=320)
            with gr.Row():
                atom = gr.Textbox(value="Person", label="Person 原子", scale=1)
                optimizer = gr.Radio(["cma", "greedy"], value="cma", label="优化器", scale=1)
                style = gr.Radio(["auto", "real", "anime", "pixel"], value="auto",
                                 label="打分风格(anime 素材务必选 anime/auto)", scale=2)
                c2f = gr.Checkbox(value=True, label="粗→细两阶段(先轮廓后五官)", scale=1)
                iters = gr.Slider(10, 400, value=60, step=10, label="评估预算", scale=2)
                width = gr.Slider(256, 1024, value=512, step=64, label="截图宽度", scale=1)
            groups = gr.CheckboxGroup(sorted(FACE_MORPH_GROUPS),
                                      value=sorted(FACE_MORPH_GROUPS),
                                      label="morph 分组(留空=全部;建议先 skull+jaw 粗调)")
            fit_btn = gr.Button("开始拟合", variant="primary")
            fit_status = gr.Markdown("")
            fit_plot = gr.Plot(label="分数曲线")
            fit_vap = gr.File(label="导出的 .vap 预设")
            fit_btn.click(run_fit, [host, port, photo, atom, optimizer, style, c2f, iters, groups, width],
                          [photo, current, fit_status, fit_plot, fit_vap])

        with gr.Tab("手动微调"):
            gr.Markdown("拟合收尾用:拖滑块**实时**写进 VaM(松手生效),满意了导出 .vap。"
                        "先点\"读取当前值\"和 VaM 对齐,免得滑块骗你。")
            with gr.Row():
                tune_atom = gr.Textbox(value="Person", label="Person 原子", scale=1)
                tune_load_btn = gr.Button("读取当前值", scale=1)
                tune_zero_btn = gr.Button("全部归零", scale=1)
                tune_shot_btn = gr.Button("刷新截图", scale=1)
            tune_status = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=3):
                    tune_sliders = []
                    for gname in sorted(FACE_MORPH_GROUPS):
                        with gr.Accordion(gname, open=(gname in ("skull", "jaw"))):
                            for mname in FACE_MORPH_GROUPS[gname]:
                                lo, hi = default_bounds(mname)
                                sl = gr.Slider(lo, hi, value=0.0, step=0.01,
                                               label=mname)
                                sl.release(
                                    (lambda n: lambda host, port, atom, v:
                                        tune_set_one(n, host, port, atom, v))(mname),
                                    [host, port, tune_atom, sl], tune_status)
                                tune_sliders.append(sl)
                with gr.Column(scale=2):
                    tune_img = gr.Image(label="VaM 当前渲染", height=420)
                    tune_export_btn = gr.Button("导出为 .vap", variant="primary")
                    tune_vap_out = gr.File(label="导出的 .vap")
                    tune_vap_in = gr.File(label="载入 .vap(应用到 VaM+滑块)",
                                          type="filepath", file_types=[".vap"])
                    tune_apply_btn = gr.Button("应用该 .vap")
            tune_load_btn.click(tune_load, [host, port, tune_atom],
                                tune_sliders + [tune_status])
            tune_zero_btn.click(tune_zero, [host, port, tune_atom],
                                tune_sliders + [tune_status])
            tune_shot_btn.click(tune_shot, [host, port, tune_atom],
                                [tune_img, tune_status])
            tune_export_btn.click(tune_export, tune_sliders,
                                  [tune_vap_out, tune_status])
            tune_apply_btn.click(tune_apply_vap,
                                 [host, port, tune_atom, tune_vap_in],
                                 tune_sliders + [tune_status])

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

            gr.Markdown("### 更新插件到 VaM\n"
                        f"把随包发行的 VamFaceBridge.cs(v{plugin_source_version() or '?'})"
                        "写进 VaM 目录,旧文件自动备份。装完在 VaM 里 reload 一次。")
            with gr.Row():
                vam_dir = gr.Textbox(value=load_config().get("vam_dir", ""),
                                     label="VaM 安装目录", scale=3,
                                     placeholder=r"例: D:\\VaM")
                inst_btn = gr.Button("更新插件", variant="primary", scale=1)
            inst_out = gr.Markdown("")
            inst_btn.click(do_install_plugin, [vam_dir], inst_out)

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()
