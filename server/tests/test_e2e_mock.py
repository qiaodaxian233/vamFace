"""端到端回归:隐藏目标拟合 —— 本轮最硬的一条测试。

流程:MockVamServer 里藏一个随机 morph 向量 → 渲出目标图 →
vamface 的拟合循环(BridgeClient + PixelScorer + 优化器)从零开始拟合 →
断言:最终分数显著高于初始分数,且导出的 .vap 合法。

这条通了,意味着 桥接客户端 / 协议 / 截图解码 / 打分 / 优化器 / .vap 导出
整条链在没有真 VaM 的机器上被持续验证 —— 真机验证只剩"API 名对不对账"。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from vamface_mcp.bridge_client import BridgeClient
from vamface_mcp.fitting import FitConfig, fit_face
from vamface_mcp.mock_vam import FaceRenderer, MockVamServer, make_hidden_target
from vamface_mcp.morph_presets import default_bounds
from vamface_mcp.scorers import PixelScorer
from vamface_mcp.vap import read_vap, write_vap


@pytest.fixture()
def mock():
    srv = MockVamServer(port=0)
    srv.start()
    yield srv
    srv.stop()


def _target_png(tmp_path: Path, hidden: dict) -> Path:
    p = tmp_path / "target.png"
    FaceRenderer().render(hidden).save(p)
    return p


def test_hidden_target_fit_improves_score(mock, tmp_path):
    hidden = make_hidden_target(seed=7, n_morphs=6, amplitude=0.9)
    target = _target_png(tmp_path, hidden)
    names = sorted(hidden)  # 只在有答案的维度上搜,保证测试快而确定

    bridge = BridgeClient("127.0.0.1", mock.port, timeout=15.0)
    cfg = FitConfig(morph_names=names,
                    bounds={n: default_bounds(n) for n in names},
                    max_iters=120, screenshot_width=256)

    # 初始分:零 morph 相对目标的像素相似度
    from vamface_mcp.fitting import decode_png_b64, load_image
    scorer = PixelScorer()
    t_img = load_image(str(target))
    base = scorer.score(
        t_img, decode_png_b64(bridge.screenshot(max_width=256)["png_base64"]))

    result = fit_face(bridge, str(target), cfg,
                      optimizer="greedy", style="pixel")
    bridge.close()

    assert result.style == "pixel"
    # 尺度无关的断言:残余误差 (1-score) 至少要减半
    assert (1 - result.best_score) < 0.5 * (1 - base), \
        f"拟合没把误差压下去: base={base:.4f} best={result.best_score:.4f}"
    assert result.best_score > 0.7  # 同域像素打分,像样的拟合应该到得了
    # 历史应单调可累积出最优(顺带验证 on_eval 计数路径没炸)
    assert len(result.history) > 10


def test_fit_moves_morphs_toward_hidden_values(mock, tmp_path):
    """方向性:大幅度的隐藏 morph,拟合值符号应与答案一致(至少多数)。"""
    hidden = {"Head Big": 1.0, "Eyes Size": 1.0, "Mouth Width": -1.0}
    target = _target_png(tmp_path, hidden)
    names = sorted(hidden)

    bridge = BridgeClient("127.0.0.1", mock.port, timeout=15.0)
    cfg = FitConfig(morph_names=names,
                    bounds={n: (-1.5, 1.5) for n in names},
                    max_iters=90, screenshot_width=256)
    result = fit_face(bridge, str(target), cfg, optimizer="greedy", style="pixel")
    bridge.close()

    agree = sum(1 for n in names
                if result.best_morphs[n] * hidden[n] > 0)
    assert agree >= 2, f"方向对齐太差: {result.best_morphs} vs {hidden}"


def test_cma_optimizer_runs_through_mock(mock, tmp_path):
    """CMA-ES 路径的冒烟(cma 已装):小预算跑通不炸即可。"""
    pytest.importorskip("cma")
    hidden = make_hidden_target(seed=3, n_morphs=4)
    target = _target_png(tmp_path, hidden)
    names = sorted(hidden)
    bridge = BridgeClient("127.0.0.1", mock.port, timeout=15.0)
    cfg = FitConfig(morph_names=names,
                    bounds={n: default_bounds(n) for n in names},
                    max_iters=40, screenshot_width=192)
    result = fit_face(bridge, str(target), cfg, optimizer="cma", style="pixel")
    bridge.close()
    assert result.best_score > 0.0
    assert set(result.best_morphs) == set(names)


def test_vap_export_roundtrip(mock, tmp_path):
    """拟合产物写成 .vap 再读回,morph 值一致(离线层与拟合层的接缝)。"""
    hidden = make_hidden_target(seed=11, n_morphs=5)
    target = _target_png(tmp_path, hidden)
    names = sorted(hidden)
    bridge = BridgeClient("127.0.0.1", mock.port, timeout=15.0)
    cfg = FitConfig(morph_names=names,
                    bounds={n: default_bounds(n) for n in names},
                    max_iters=30, screenshot_width=192)
    result = fit_face(bridge, str(target), cfg, optimizer="greedy", style="pixel")
    bridge.close()

    vap = tmp_path / "fit.vap"
    write_vap(vap, result.best_morphs)
    back = read_vap(vap)
    for n in names:
        assert abs(back[n] - result.best_morphs[n]) < 1e-6
    # .vap 是合法 JSON 且带 VaM 结构
    doc = json.loads(vap.read_text(encoding="utf-8"))
    assert "storables" in doc


def test_mock_cli_seed_mode(tmp_path, monkeypatch, capsys):
    """vamface-mock --seed 的目标图生成路径(不真的常驻监听)。"""
    from vamface_mcp import mock_vam

    monkeypatch.chdir(tmp_path)

    # 让 serve 阶段立刻返回
    def _boom():
        raise KeyboardInterrupt

    monkeypatch.setattr(mock_vam, "_serve_wait", _boom)
    rc = mock_vam.main(["--port", "0", "--seed", "42"])
    assert rc == 0
    assert (tmp_path / "mock_target_42.png").is_file()
    out = capsys.readouterr().out
    assert "标准答案" in out
