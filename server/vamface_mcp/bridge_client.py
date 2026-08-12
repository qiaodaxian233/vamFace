"""TCP client for the VamFaceBridge plugin running inside VaM.

Protocol: newline-delimited JSON over TCP (see docs/protocol.md).
Request : {"id": "<n>", "cmd": "<name>", "args": {...}}
Response: {"id": "<n>", "ok": true, "data": {...}}
        | {"id": "<n>", "ok": false, "error": "..."}

Responses are sent in request order per connection, but we still match by
id to be safe. A single BridgeClient is not thread-safe by itself; all
calls are serialized behind a lock, which is fine for MCP usage.
"""

from __future__ import annotations

import itertools
import json
import socket
import threading
from typing import Any, Dict, List, Optional


class BridgeError(RuntimeError):
    """Raised when the bridge returns ok=false or the connection breaks."""


def check_handshake(ping_data: Dict[str, Any]) -> Optional[str]:
    """检查 ping 返回的协议版本,不匹配/缺失时返回人话警告,匹配返回 None。"""
    from . import PROTOCOL_VERSION, __version__

    proto = ping_data.get("protocol")
    ver = ping_data.get("version", "?")
    if proto is None:
        return (f"插件 v{ver} 没报协议版本(0.5 之前的版本)——server 是 "
                f"v{__version__}/协议{PROTOCOL_VERSION}。建议更新插件:GUI 连接调试页"
                f"一键更新,或手动复制 plugin/VamFaceBridge.cs 后在 VaM 里 reload。")
    if int(proto) != PROTOCOL_VERSION:
        return (f"协议版本不匹配:插件 v{ver}/协议{proto},server v{__version__}/"
                f"协议{PROTOCOL_VERSION}。两边有一边旧了,部分命令可能对不上,"
                f"请把插件和 server 更到同一版本。")
    return None


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8787,
                 timeout: float = 30.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._file = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # -- connection management ------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as e:
            raise BridgeError(
                f"cannot connect to VamFaceBridge at {self.host}:{self.port} — "
                f"is VaM running with the session plugin loaded? ({e})"
            ) from e
        sock.settimeout(self.timeout)
        self._sock = sock
        self._file = sock.makefile("rb")

    def close(self) -> None:
        try:
            if self._file is not None:
                self._file.close()
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        self._file = None

    # -- request/response -----------------------------------------------------

    def call(self, cmd: str, timeout: Optional[float] = None, **args: Any) -> Dict[str, Any]:
        """Send one command and return the `data` object of the response."""
        with self._lock:
            self._ensure_connected()
            assert self._sock is not None and self._file is not None
            if timeout is not None:
                self._sock.settimeout(timeout)
            rid = str(next(self._ids))
            payload = json.dumps({"id": rid, "cmd": cmd, "args": args})
            try:
                self._sock.sendall(payload.encode("utf-8") + b"\n")
                while True:
                    line = self._file.readline()
                    if not line:
                        self.close()
                        raise BridgeError("connection closed by VaM")
                    try:
                        resp = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                        raise BridgeError(f"bad response from bridge: {e}") from e
                    if str(resp.get("id", "")) != rid:
                        continue  # stale response from a previous timeout
                    if not resp.get("ok"):
                        raise BridgeError(resp.get("error", "unknown bridge error"))
                    return resp.get("data") or {}
            except socket.timeout as e:
                self.close()
                raise BridgeError(f"bridge call '{cmd}' timed out") from e
            except OSError as e:
                self.close()
                raise BridgeError(f"bridge I/O error on '{cmd}': {e}") from e
            finally:
                if timeout is not None and self._sock is not None:
                    self._sock.settimeout(self.timeout)

    # -- convenience wrappers ---------------------------------------------------

    def ping(self) -> Dict[str, Any]:
        return self.call("ping")

    def handshake(self) -> Dict[str, Any]:
        """ping + 协议版本对账。返回 ping 的 data,可能追加 compat_warning。

        对不上**只警告不阻塞**(教训5:握手失败大概率是旧插件,功能多半还能用,
        由用户决定要不要升级)。
        """
        info = self.ping()
        w = check_handshake(info)
        if w:
            info = dict(info)
            info["compat_warning"] = w
        return info

    def list_atoms(self) -> Dict[str, Any]:
        return self.call("list_atoms")

    def list_morphs(self, atom: str, filter: str = "", region: str = "",
                    limit: int = 200) -> Dict[str, Any]:
        return self.call("list_morphs", atom=atom, filter=filter,
                         region=region, limit=limit)

    def get_morphs(self, atom: str, changed_only: bool = True) -> Dict[str, float]:
        data = self.call("get_morphs", atom=atom, changed_only=changed_only)
        return {k: float(v) for k, v in (data.get("values") or {}).items()}

    def set_morphs(self, atom: str, values: Dict[str, float],
                   clamp: bool = True) -> Dict[str, Any]:
        return self.call("set_morphs", atom=atom, values=values, clamp=clamp)

    def reset_morphs(self, atom: str) -> Dict[str, Any]:
        return self.call("reset_morphs", atom=atom)

    def load_scene(self, path: str) -> Dict[str, Any]:
        return self.call("load_scene", path=path, timeout=120.0)

    def focus_head(self, atom: str) -> Dict[str, Any]:
        return self.call("focus_head", atom=atom)

    def screenshot(self, max_width: int = 0,
                   camera: Optional[bool] = None) -> Dict[str, Any]:
        """Returns {"width", "height", "method", "png_base64"}; can be several MB.

        camera=None 沿用插件默认(0.5.5 起为相机 RTT,无 UI);False 强制
        旧的整屏捕获(调试/对照用);老插件会忽略这个参数。
        """
        kw: Dict[str, Any] = {"max_width": max_width, "timeout": 60.0}
        if camera is not None:
            kw["camera"] = bool(camera)
        return self.call("screenshot", **kw)

    # -- generic storable/param access ---------------------------------------

    def list_storables(self, atom: str) -> List[str]:
        return list(self.call("list_storables", atom=atom).get("storables") or [])

    def list_params(self, atom: str, storable: str) -> Dict[str, List[str]]:
        return self.call("list_params", atom=atom, storable=storable)

    def get_param(self, atom: str, storable: str, param: str) -> Dict[str, Any]:
        return self.call("get_param", atom=atom, storable=storable, param=param)

    def set_param(self, atom: str, storable: str, param: str, value: Any,
                  type: str = "") -> Dict[str, Any]:
        return self.call("set_param", atom=atom, storable=storable,
                         param=param, value=value, type=type)

    def call_action(self, atom: str, storable: str, action: str) -> Dict[str, Any]:
        return self.call("call_action", atom=atom, storable=storable, action=action)

    def list_characters(self, atom: str) -> List[str]:
        return list(self.call("list_characters", atom=atom).get("characters") or [])

    def set_character(self, atom: str, name: str) -> Dict[str, Any]:
        return self.call("set_character", atom=atom, name=name, timeout=120.0)

    def find_color_params(self, atom: str,
                          storable_filter: str = "skin") -> List[Dict[str, str]]:
        """Scan storables and return [{storable, param}] for color params.

        `storable_filter` is a case-insensitive substring; empty scans all.
        """
        hits: List[Dict[str, str]] = []
        needle = storable_filter.lower()
        for sid in self.list_storables(atom):
            if needle and needle not in sid.lower():
                continue
            try:
                params = self.list_params(atom, sid)
            except BridgeError:
                continue
            for p in params.get("colors") or []:
                hits.append({"storable": sid, "param": p})
        return hits
