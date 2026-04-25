from __future__ import annotations

import os
import socket


DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 5566


def send_udp_message(message: str, host: str | None = None, port: int | None = None) -> None:
    address = (host or os.getenv("ENVTEST_UDP_HOST", DEFAULT_UDP_HOST), int(port or os.getenv("ENVTEST_UDP_PORT", DEFAULT_UDP_PORT)))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode("utf-8"), address)
    finally:
        sock.close()
#向现有EnvTest UDP server发送控制文本，复用原控制链


def send_stop_message(host: str | None = None, port: int | None = None) -> None:
    send_udp_message("start=0", host=host, port=port)
#发送统一停止命令，动作完成或超时后让player回到待机
