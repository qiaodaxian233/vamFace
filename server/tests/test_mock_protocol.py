"""协议层测试:用真 socket + BridgeClient 打 MockVamServer,逐命令对账。

这组测试同时锁死两件事:
  1. mock 对 docs/protocol.md 的实现是对的;
  2. BridgeClient 对协议的理解是对的(以前只能靠真机验证)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from vamface_mcp.bridge_client import BridgeClient, BridgeError
from vamface_mcp.mock_vam import MORPH_DEFS, MockVamServer


@pytest.fixture()
def bridge():
    srv = MockVamServer(port=0)  # 临时端口,测试可并行
    srv.start()
    b = BridgeClient("127.0.0.1", srv.port, timeout=10.0)
    yield b
    b.close()
    srv.stop()


def test_ping(bridge):
    info = bridge.ping()
    assert info["app"] == "MockVaM"
    assert info["version"].startswith("mock")


def test_list_atoms(bridge):
    atoms = bridge.list_atoms()["atoms"]
    assert {"uid": "Person", "type": "Person"} in atoms


def test_list_morphs_filter_and_region(bridge):
    data = bridge.list_morphs("Person", filter="nose")
    names = [m["name"] for m in data["morphs"]]
    assert names and all("nose" in n.lower() for n in names)
    data = bridge.list_morphs("Person", region="eyes")
    assert all(m["region"] == "eyes" for m in data["morphs"])


def test_set_get_morphs_and_missing(bridge):
    # 存在的名字被应用,不存在的进 missing —— 教训5,不许报错
    r = bridge.set_morphs("Person", {"Nose Width": 0.5, "不存在的morph": 1.0})
    assert r["applied"] == 1
    assert r["missing"] == ["不存在的morph"]
    vals = bridge.get_morphs("Person", changed_only=True)
    assert vals == {"Nose Width": 0.5}


def test_set_morphs_clamp(bridge):
    bridge.set_morphs("Person", {"Nose Width": 99.0}, clamp=True)
    _, lo, hi = MORPH_DEFS["Nose Width"]
    assert bridge.get_morphs("Person")["Nose Width"] == hi


def test_reset_morphs(bridge):
    bridge.set_morphs("Person", {"Jaw Width": 0.7})
    r = bridge.reset_morphs("Person")
    assert r["reset"] >= 1
    assert bridge.get_morphs("Person") == {}


def test_bad_atom_is_error(bridge):
    with pytest.raises(BridgeError, match="atom not found"):
        bridge.set_morphs("Ghost", {"Nose Width": 0.1})


def test_screenshot_shape_and_downscale(bridge):
    shot = bridge.screenshot(max_width=128)
    assert shot["width"] == 128
    from vamface_mcp.fitting import decode_png_b64
    img = decode_png_b64(shot["png_base64"])
    assert img.shape[1] == 128 and img.shape[2] == 3


def test_screenshot_changes_when_morphs_change(bridge):
    from vamface_mcp.fitting import decode_png_b64
    a = decode_png_b64(bridge.screenshot(max_width=128)["png_base64"])
    bridge.set_morphs("Person", {"Head Big": 1.0, "Eyes Size": 1.0})
    b = decode_png_b64(bridge.screenshot(max_width=128)["png_base64"])
    assert (a != b).any()


def test_storable_params_roundtrip(bridge):
    assert "skin" in bridge.list_storables("Person")
    params = bridge.list_params("Person", "skin")
    assert "Skin Color" in params["colors"]
    bridge.set_param("Person", "skin", "Gloss", 0.9)
    assert bridge.get_param("Person", "skin", "Gloss")["value"] == 0.9
    with pytest.raises(BridgeError, match="param not found"):
        bridge.get_param("Person", "skin", "没有的参数")


def test_skin_color_actually_changes_render(bridge):
    """写 Skin Color 必须影响渲染 —— 皮肤 L0 流水线的 e2e 依据。"""
    from vamface_mcp.fitting import decode_png_b64
    a = decode_png_b64(bridge.screenshot(max_width=128)["png_base64"])
    bridge.set_param("Person", "skin", "Skin Color",
                     {"h": 0.6, "s": 0.5, "v": 0.5})  # 明显偏蓝的假肤色
    b = decode_png_b64(bridge.screenshot(max_width=128)["png_base64"])
    assert (a != b).any()


def test_characters(bridge):
    chars = bridge.list_characters("Person")
    assert len(chars) >= 2
    r = bridge.set_character("Person", chars[1])
    assert r["selected"] == chars[1]
    with pytest.raises(BridgeError, match="character not found"):
        bridge.set_character("Person", "Nobody")
