"""更新检查 + 一键把插件装进 VaM 目录。

两件事:
  1. check_latest()  — 拉 GitHub 上 main 分支的 __init__.py,比对版本号。
     任何网络/解析失败都**降级成 error 字段**,不抛异常不阻塞(教训5)。
  2. install_plugin(vam_dir) — 把随包发行的 VamFaceBridge.cs 复制到
     (VaM)/Custom/Scripts/VamFace/,旧文件先备份成 .bak。之后仍需在 VaM 里
     对 session plugin 点一次 reload —— 这步没有 API,只能人做。

插件源文件的查找顺序:
  a) 包内资源 vamface_mcp/resources/VamFaceBridge.cs(pip 安装也在)
  b) 仓库布局 ../../plugin/VamFaceBridge.cs(editable 安装的兜底)
两份文件由 tests/test_v05_features.py 强制逐字节一致 —— 改插件只改
plugin/ 下那份,然后跑 python scripts 同步?不,直接 cp 再跑测试,测试红
就是忘了同步(教训6:同一份内容存两处必须有 gate)。

VaM 目录记在 ~/.vamface/config.json,填一次就够。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import __version__

REPO_RAW_INIT = ("https://raw.githubusercontent.com/qiaodaxian233/vamFace/"
                 "main/server/vamface_mcp/__init__.py")
CONFIG_PATH = Path.home() / ".vamface" / "config.json"
PLUGIN_REL_DEST = Path("Custom") / "Scripts" / "VamFace" / "VamFaceBridge.cs"


# ---------------------------------------------------------------------------
# 版本检查
# ---------------------------------------------------------------------------

def _default_fetch(url: str, timeout: float = 6.0) -> str:
    from urllib.request import urlopen

    with urlopen(url, timeout=timeout) as r:  # noqa: S310 (固定 https 常量)
        return r.read().decode("utf-8", errors="replace")


def _parse_version(text: str) -> Optional[str]:
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else None


def _ver_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4]) or (0,)


def check_latest(fetch: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """比对本地与 GitHub main 的版本。永不抛异常。

    返回 {installed, latest, update_available, error}。
    latest/update_available 在失败时为 None,error 带原因。
    """
    out: Dict[str, Any] = {"installed": __version__, "latest": None,
                           "update_available": None, "error": None}
    try:
        text = (fetch or _default_fetch)(REPO_RAW_INIT)
        latest = _parse_version(text)
        if latest is None:
            out["error"] = "拉到了文件但没解析出 __version__(仓库结构变了?)"
            return out
        out["latest"] = latest
        out["update_available"] = _ver_tuple(latest) > _ver_tuple(__version__)
    except Exception as e:
        out["error"] = f"检查更新失败(离线/GitHub 不可达都会这样,不影响使用): {e}"
    return out


# ---------------------------------------------------------------------------
# 插件安装
# ---------------------------------------------------------------------------

def plugin_source_path() -> Optional[Path]:
    """定位随包发行的插件源码,找不到返回 None(不抛)。"""
    pkg = Path(__file__).resolve().parent
    candidates = [
        pkg / "resources" / "VamFaceBridge.cs",          # 包内资源
        pkg.parent.parent / "plugin" / "VamFaceBridge.cs",  # 仓库布局兜底
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def plugin_source_version() -> Optional[str]:
    src = plugin_source_path()
    if src is None:
        return None
    m = re.search(r'VERSION\s*=\s*"([^"]+)"', src.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def install_plugin(vam_dir: str) -> Dict[str, Any]:
    """把插件复制进 VaM 目录。返回 {ok, dest, backup, error, note}。

    只做文件层面的校验(目录存在),不硬性要求 VaM.exe 在场 —— 用户可能给的
    是网络盘/移动过的安装目录,硬 gate 只会挡住正当用法(教训5)。看起来不像
    VaM 目录时在 note 里提醒,但照样装。
    """
    out: Dict[str, Any] = {"ok": False, "dest": None, "backup": None,
                           "error": None, "note": None}
    src = plugin_source_path()
    if src is None:
        out["error"] = ("找不到插件源码(包里没带 resources/VamFaceBridge.cs,"
                        "仓库布局下也没有 plugin/)。重装 pip 包或从仓库跑。")
        return out
    base = Path(vam_dir).expanduser()
    if not base.is_dir():
        out["error"] = f"目录不存在: {base}"
        return out
    if not (base / "VaM.exe").exists() and not (base / "Custom").is_dir():
        out["note"] = ("提醒:这里既没有 VaM.exe 也没有 Custom/,看起来不像 VaM "
                       "安装目录 —— 仍已按你给的路径安装。")
    dest = base / PLUGIN_REL_DEST
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            backup = dest.with_suffix(f".cs.bak-{int(time.time())}")
            dest.replace(backup)
            out["backup"] = str(backup)
        dest.write_bytes(src.read_bytes())
        out["ok"] = True
        out["dest"] = str(dest)
    except OSError as e:
        out["error"] = f"复制失败: {e}"
    return out


# ---------------------------------------------------------------------------
# 配置持久化(目前只有 vam_dir)
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(**updates: Any) -> Dict[str, Any]:
    cfg = load_config()
    cfg.update(updates)
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    except OSError:
        pass  # 存不了就每次手填,不阻塞
    return cfg
