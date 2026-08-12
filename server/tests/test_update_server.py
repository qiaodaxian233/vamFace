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


# ---------------------------------------------------------------------------
# v0.6.1:git 快路自愈(zip 覆盖残留把 pull 挡死的自锁)
# ---------------------------------------------------------------------------

_GIT_BLOCK_ERR = """error: The following untracked working tree files would be overwritten by merge:
\tserver/vamface_mcp/updater.py
\tscripts/install_anime_deps.bat
Please move or remove them before you merge.
Aborting
"""


class _R:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_untracked_blocker_parser():
    got = updater._untracked_blockers(_GIT_BLOCK_ERR)
    assert got == ["server/vamface_mcp/updater.py",
                   "scripts/install_anime_deps.bat"]
    assert updater._untracked_blockers("fatal: not a git repository") == []


def test_git_self_heals_untracked_blockers(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    (root / ".git").mkdir()
    (root / "server" / "vamface_mcp").mkdir()
    (root / "server" / "vamface_mcp" / "updater.py").write_text("stale zip copy")
    (root / "scripts").mkdir()
    (root / "scripts" / "install_anime_deps.bat").write_text("stale")
    monkeypatch.setattr(updater, "repo_root", lambda: root)

    pulls = {"n": 0}

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "rev-parse"]:
            return _R(0, out="aaa\n" if pulls["n"] < 2 else "bbb\n")
        if cmd[:2] == ["git", "pull"]:
            pulls["n"] += 1
            if pulls["n"] == 1:
                return _R(1, err=_GIT_BLOCK_ERR)  # 第一次被残留挡下
            return _R(0, out="Fast-forward")       # 清障后重试成功
        return _R(0)

    r = updater.update_server(runner=fake_run)
    assert r["ok"] and r["method"] == "git"
    assert "清障 2 个" in r["detail"]
    # 原件挪去 .bak,不删
    assert (root / "server/vamface_mcp/updater.py.pre-update.bak").read_text() \
        == "stale zip copy"
    assert not (root / "server/vamface_mcp/updater.py").exists()
    assert pulls["n"] == 2


def test_git_heal_refuses_to_touch_unmanaged_files(tmp_path, monkeypatch):
    """挡路的是前缀外的用户文件 → 一根手指不碰,老老实实降级 zip。"""
    root = _fake_repo(tmp_path)
    (root / ".git").mkdir()
    mine = root / "my_secret_notes.txt"
    mine.write_text("precious")
    monkeypatch.setattr(updater, "repo_root", lambda: root)

    err = ("error: The following untracked working tree files would be "
           "overwritten by merge:\n\tmy_secret_notes.txt\nAborting\n")

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "rev-parse"]:
            return _R(0, out="aaa\n")
        return _R(1, err=err)

    blob = _make_zip({"server/vamface_mcp/__init__.py": "__version__ = 'x'"})
    r = updater.update_server(fetch_bytes=lambda url: blob, runner=fake_run)
    assert r["ok"] and r["method"] == "zip"
    assert mine.read_text() == "precious"          # 没被动
    assert not mine.with_name(mine.name + ".pre-update.bak").exists()
