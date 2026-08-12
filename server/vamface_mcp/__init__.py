"""VamFace MCP: automate Virt-A-Mate face creation from photos."""

# 单一版本源:pyproject 用 setuptools dynamic 从这里读,别再在两处写版本号。
__version__ = "0.7.2"

# 桥接协议版本:server 与插件(及 mock)在 ping 时对账。
# 插件端是 plugin/VamFaceBridge.cs 里的 PROTOCOL const,改协议时两边同步 +1。
PROTOCOL_VERSION = 1
