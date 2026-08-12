"""v0.5:版本握手 / 截图缓存 / 更新器 / 插件双份同步。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from vamface_mcp import PROTOCOL_VERSION, __version__
from vamface_mcp.bridge_client import BridgeClient, check_handshake
from vamface_mcp.fitting import FitConfig, make_evaluator
from vamface_mcp.mock_vam import MockVamServer
from vamface_mcp.scorers import PixelScorer
from vamface_mcp import updater

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 握手
# ---------------------------------------------------------------------------

def test_check_handshake_match_is_silent():
    assert check_handshake({"version": "x", "protocol": PROTOCOL_VERSION}) is None


def test_check_handshake_missing_protocol_warns_not_raises():
    w = check_handshake({"version": "0.4.0"})
    assert w and "协议" in w


def test_check_handshake_mismatch_warns():
    w = check_handshake({"version": "9", "protocol": PROTOCOL_VERSION + 1})
    assert w and str(PROTOCOL_VERSION) in w


def test_mock_handshake_end_to_end():
    srv = MockVamServer(port=0)
    srv.start()
    try:
        c = BridgeClient("127.0.0.1", srv.port)
        info = c.handshake()
        assert info["protocol"] == PROTOCOL_VERSION
        assert "compat_warning" not in info
        c.close()
    finally:
        srv.stop()


# ---------------------------------------------------------------------------
# 截图缓存
# ---------------------------------------------------------------------------

class CountingBridge:
    """最小桥接桩:统计 screenshot 次数,渲染 = 常数图。"""

    def __init__(self):
        self.shots = 0
        import base64
        import io as _io
        from PIL import Image
        buf = _io.BytesIO()
        Image.new("RGB", (32, 32), (128, 128, 128)).save(buf, format="PNG")
        self._png = base64.b64encode(buf.getvalue()).decode("ascii")

    def set_morphs(self, atom, values, clamp=True):
        return {"applied": len(values), "missing": []}

    def screenshot(self, max_width=0):
        self.shots += 1
        return {"width": 32, "height": 32, "png_base64": self._png}


def _evaluator(use_cache=True):
    bridge = CountingBridge()
    target = np.zeros((32, 32, 3), dtype=np.uint8)
    cfg = FitConfig(morph_names=["A", "B"], use_cache=use_cache)
    ev = make_evaluator(bridge, target, PixelScorer(), cfg)
    return bridge, ev


def test_cache_dedups_exact_revisit():
    bridge, ev = _evaluator()
    v = {"A": 0.5, "B": -0.25}
    s1 = ev(v)
    s2 = ev(dict(v))
    assert s1 == s2
    assert bridge.shots == 1
    assert ev.cache_hits == 1


def test_cache_quantization_treats_1e4_as_distinct():
    bridge, ev = _evaluator()
    ev({"A": 0.5000})
    ev({"A": 0.5001})  # 1e-4 粒度上不同 → 必须真评估
    assert bridge.shots == 2


def test_cache_epoch_invalidates():
    bridge, ev = _evaluator()
    v = {"A": 0.5}
    ev(v)
    ev.bump_epoch()  # 模拟阶段冻结绕过 evaluate 写了状态
    ev(v)
    assert bridge.shots == 2
    assert ev.cache_hits == 0


def test_cache_can_be_disabled():
    bridge, ev = _evaluator(use_cache=False)
    v = {"A": 0.5}
    ev(v)
    ev(v)
    assert bridge.shots == 2


# ---------------------------------------------------------------------------
# 更新器
# ---------------------------------------------------------------------------

def test_check_latest_update_available():
    r = updater.check_latest(fetch=lambda url: '__version__ = "99.0.0"')
    assert r["latest"] == "99.0.0" and r["update_available"] is True
    assert r["error"] is None


def test_check_latest_same_version():
    r = updater.check_latest(fetch=lambda url: f'__version__ = "{__version__}"')
    assert r["update_available"] is False


def test_check_latest_network_failure_degrades():
    def boom(url):
        raise OSError("no network")
    r = updater.check_latest(fetch=boom)
    assert r["error"] and r["latest"] is None  # 不抛异常(教训5)


def test_install_plugin_copies_and_backs_up(tmp_path):
    vam = tmp_path / "VaM"
    (vam / "Custom").mkdir(parents=True)
    r1 = updater.install_plugin(str(vam))
    assert r1["ok"], r1
    dest = Path(r1["dest"])
    assert dest.read_bytes() == (REPO / "plugin" / "VamFaceBridge.cs").read_bytes()
    dest.write_text("old", encoding="utf-8")  # 装旧文件,再装应备份
    r2 = updater.install_plugin(str(vam))
    assert r2["ok"] and r2["backup"]
    assert Path(r2["backup"]).read_text(encoding="utf-8") == "old"


def test_install_plugin_bad_dir_degrades(tmp_path):
    r = updater.install_plugin(str(tmp_path / "nope"))
    assert not r["ok"] and r["error"]


def test_install_plugin_unlikely_dir_warns_but_installs(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    r = updater.install_plugin(str(d))
    assert r["ok"] and r["note"]  # 不硬 gate,提醒照装


# ---------------------------------------------------------------------------
# 插件双份同步(包内资源 vs 仓库 plugin/)—— 教训6 的 gate
# ---------------------------------------------------------------------------

def test_packaged_plugin_matches_repo_copy():
    repo_cs = REPO / "plugin" / "VamFaceBridge.cs"
    pkg_cs = REPO / "server" / "vamface_mcp" / "resources" / "VamFaceBridge.cs"
    assert pkg_cs.is_file(), "包内资源缺失:cp plugin/VamFaceBridge.cs server/vamface_mcp/resources/"
    assert repo_cs.read_bytes() == pkg_cs.read_bytes(), \
        "plugin/ 与 resources/ 的 VamFaceBridge.cs 不一致 —— 改了一处忘了同步另一处"


def test_plugin_and_server_versions_agree():
    assert updater.plugin_source_version() == __version__, \
        "插件 VERSION 与 server __version__ 不一致,发布前两边同步"
