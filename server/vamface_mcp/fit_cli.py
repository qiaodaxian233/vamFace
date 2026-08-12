"""Standalone CLI: fit the VaM face to a photo — no MCP client needed.

The MCP server is just one frontend; this is the other. It talks to the
same VamFaceBridge plugin over TCP and runs the same fitting loop.

Usage:
    vamface-fit photo.png
    vamface-fit photo.png --atom Person --optimizer cma --iters 80
    vamface-fit photo.png --groups skull jaw --optimizer greedy
    vamface-fit photo.png --out mylook.vap --host 127.0.0.1 --port 8787

Still requires VaM running with the VamFaceBridge session plugin loaded
(the loop needs live screenshots to score against). For a fully offline
path that needs no VaM at all, see docs/roadmap.md v0.3 (route A).
"""

from __future__ import annotations

# --- direct-run bootstrap -------------------------------------------------
# 包内模块被当成裸文件跑(Windows 上双击 / 直接敲文件路径)时,相对导入会炸
# "attempted relative import with no known parent package"。检测到这种情况就
# 把包父目录塞进 sys.path,以正确的模块身份重跑一遍。用户实测踩过(2026-08-12)。
if __package__ in (None, ""):
    import os as _os, sys as _sys, runpy as _runpy
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    _runpy.run_module("vamface_mcp.fit_cli", run_name="__main__")
    _sys.exit(0)
# ---------------------------------------------------------------------------


import argparse
import json
import sys
import time
from pathlib import Path

from .bridge_client import BridgeClient, BridgeError
from .fitting import FitConfig, fit_face
from .morph_presets import (FACE_MORPH_GROUPS, default_bounds,
                            default_face_morph_names)
from .vap import write_vap


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="vamface-fit",
        description="Fit the VaM face to a target photo (no MCP needed).")
    parser.add_argument("photo", help="path to the target face photo")
    parser.add_argument("--atom", default="Person", help="Person atom uid (default: Person)")
    parser.add_argument("--optimizer", choices=["cma", "greedy"], default="cma")
    parser.add_argument("--style", choices=["auto", "real", "anime", "pixel"],
                        default="auto",
                        help="打分风格: auto=探测目标图 / real=ArcFace / "
                             "anime=animeface 几何 / pixel=像素(mock 测试用)")
    parser.add_argument("--anime-onnx", default=None,
                        help="可选: anime 识别 ONNX 模型路径(style=anime 时加权合入)")
    parser.add_argument("--coarse-to-fine", action="store_true",
                        help="两阶段拟合: 先轮廓(skull/jaw/cheeks)冻结,再五官")
    parser.add_argument("--no-prior", action="store_true",
                        help="禁用几何先验种子(默认开; 需要打分器里有几何项)")
    parser.add_argument("--no-neutralize", action="store_true",
                        help="拟合前不清零表情类 morph(默认清零防作弊解)")
    parser.add_argument("--no-cache", action="store_true",
                        help="禁用截图缓存(默认开: 重访相同 morph 向量直接复用分数)")
    parser.add_argument("--iters", type=int, default=60, help="evaluation budget")
    parser.add_argument("--basis", action="store_true",
                        help="先扫整头角色 morph 当起点(v0.6 基底粗定位)")
    parser.add_argument("--groups", nargs="*", choices=sorted(FACE_MORPH_GROUPS),
                        help="morph regions to optimize (default: all)")
    parser.add_argument("--out", help="output .vap path (default: fit_<ts>.vap)")
    parser.add_argument("--width", type=int, default=512, help="screenshot max width")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)

    photo = Path(args.photo)
    if not photo.is_file():
        print(f"error: photo not found: {photo}", file=sys.stderr)
        return 2

    if args.groups:
        names = [n for g in args.groups for n in FACE_MORPH_GROUPS[g]]
    else:
        names = default_face_morph_names()

    bridge = BridgeClient(args.host, args.port)
    try:
        info = bridge.handshake()
        print(f"connected: VamFaceBridge v{info.get('version')} on {args.host}:{args.port}")
        if info.get("compat_warning"):
            print(f"WARNING: {info['compat_warning']}", file=sys.stderr)
    except BridgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    cfg = FitConfig(
        atom=args.atom,
        morph_names=names,
        bounds={n: default_bounds(n) for n in names},
        max_iters=args.iters,
        screenshot_width=args.width,
        use_cache=not args.no_cache,
    )

    print(f"fitting over {len(names)} morphs, optimizer={args.optimizer}, "
          f"budget={args.iters} evaluations ...")
    t0 = time.time()
    try:
        result = fit_face(bridge, str(photo), cfg, optimizer=args.optimizer,
                          use_basis=args.basis,
                          style=args.style, anime_onnx=args.anime_onnx,
                          use_prior=not args.no_prior,
                          neutralize=not args.no_neutralize,
                          coarse_to_fine=args.coarse_to_fine)
    except BridgeError as e:
        print(f"error during fit: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s — best score: {result.best_score:.4f} "
          f"(style={result.style}, scorer={result.scorer_name}, "
          f"stages={result.stage_count}, cache hits={result.cache_hits})")
    if result.neutralized:
        print(f"  已清零表情 morph: {', '.join(result.neutralized)}")
    if result.prior_seed:
        print(f"  先验种子: {json.dumps(result.prior_seed, ensure_ascii=False)}")
    for h in result.hints:
        print(f"  提示: {h}")
    if result.warning:
        print(f"WARNING: {result.warning}", file=sys.stderr)
        if "NullScorer" in result.warning:
            print("         (score above is a placeholder; install extras: "
                  "pip install -e '.[fit]')", file=sys.stderr)
    if result.basis:
        print(f"  基底: {json.dumps(result.basis, ensure_ascii=False)}(已含在 .vap)")
    if result.renamed:
        pairs = ", ".join(f"{k}→{v}" for k, v in sorted(result.renamed.items()))
        print(f"  别名解析生效 {len(result.renamed)} 个(概念名→实际名): {pairs}")
    if result.missing:
        print(f"  目标 VaM 缺 {len(result.missing)} 个精选 morph(未参与拟合):")
        print(f"  {', '.join(result.missing)}")

    out = args.out or f"fit_{int(time.time())}.vap"
    write_vap(out, result.best_morphs)
    print(f"saved preset: {out} ({len(result.best_morphs)} morphs)")
    print("non-zero morphs:")
    nz = {k: round(v, 3) for k, v in result.best_morphs.items() if abs(v) > 0.01}
    print(json.dumps(nz, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
