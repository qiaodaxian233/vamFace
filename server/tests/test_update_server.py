"""update_server:git 快路 + zipball 覆盖降级,只增改不删、防路径穿越。"""
import io
import subprocess
import zipfile
from pathlib import Path

from vamface_mcp import updater


def _make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, data in entries.items():
            zf.writestr(f"vamFace-main/{rel}", data)
    return buf.getvalue()


def _fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "pyproject.toml").write_text("[project]")
    (tmp_path / "plugin").mkdir()
    return tmp_path


def test_zip_overlay_writes_allowed_and_skips_others(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    blob = _make_zip({
        "server/vamface_mcp/__init__.py": "__version__ = 'x'",
        "plugin/VamFaceBridge.cs": "// cs",
        "README.md": "outside overlay prefixes",
        "../evil.txt": "path traversal",
    })
    r = updater.update_server(fetch_bytes=lambda url: blob,
                              runner=None)  # 无 .git → 直接走 zip
    assert r["ok"] and r["method"] == "zip" and r["restart_required"]
    assert (root / "server/vamface_mcp/__init__.py").read_text() == "__version__ = 'x'"
    assert (root / "plugin/VamFaceBridge.cs").exists()
    assert not (root / "README.md").exists()          # 前缀外不写
    assert not (tmp_path.parent / "evil.txt").exists()  # 穿越被拦


def test_zip_overlay_never_deletes(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    keep = root / "server" / "my_local_notes.txt"
    keep.write_text("mine")
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    blob = _make_zip({"server/vamface_mcp/a.py": "pass"})
    r = updater.update_server(fetch_bytes=lambda url: blob)
    assert r["ok"] and keep.read_text() == "mine"


def test_git_fast_path(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    (root / ".git").mkdir()
    monkeypatch.setattr(updater, "repo_root", lambda: root)

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = {"rev-parse": "abc1234\n"}
        if "pull" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="Already up to date.\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=out["rev-parse"], stderr="")

    import shutil
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")
    r = updater.update_server(runner=fake_run)
    assert r["ok"] and r["method"] == "git"
    assert r["restart_required"] is False  # 前后 HEAD 相同 = 没变


def test_no_repo_root_fails_gracefully(monkeypatch):
    monkeypatch.setattr(updater, "repo_root", lambda: None)
    r = updater.update_server()
    assert not r["ok"] and "仓库根" in r["error"]
