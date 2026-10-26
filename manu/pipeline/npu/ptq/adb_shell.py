#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
adb_shell.py -- a minimal, dependency-free ADB client that can run ONE shell
command on a device that is reachable over TCP (port 5555).

WHY THIS EXISTS
---------------
The board's sshd stopped answering in the kex phase ("Connection reset by
peer") while the board itself still pings, and the board's ADB daemon is
listening on 5555.  The workstation has no `adb` binary and no passwordless
sudo to install one, so this script speaks the ADB wire protocol directly.

The protocol (all integers little endian, 24-byte header):

    uint32 command       'CNXN' 0x4e584e43 | 'OPEN' 0x4e584f50
                         'OKAY' 0x59414b4f | 'WRTE' 0x45545257
                         'CLSE' 0x45534c43
    uint32 arg0          CNXN: protocol version (0x01000000)
                         OPEN: local id
    uint32 arg1          CNXN: max payload (0x1000000)
                         OPEN: 0
    uint32 data_length   payload bytes that follow the header
    uint32 data_crc32    CRC32 of the payload
    uint32 magic         command ^ 0xffffffff

Handshake: the host sends CNXN with the payload
"host::features=shell_v2\0"; the device answers with its own CNXN.  Then the
host sends OPEN with the payload "shell:<command>\0"; the device answers
OKAY and streams the output as WRTE packets until it sends CLSE.

Usage
-----
    python3 adb_shell.py 192.168.0.64 5555 "ls -l /root"
    python3 adb_shell.py 192.168.0.64 5555 "pkill -9 -x rknn_heatmap_sm"
"""

from __future__ import annotations

import socket
import struct
import sys
import zlib

CNXN = 0x4E584E43
OPEN = 0x4E584F50
OKAY = 0x59414B4F
WRTE = 0x45545257
CLSE = 0x45534C43
MAX_PAYLOAD = 0x100000
VERSION = 0x01000000


def pack(command: int, arg0: int, arg1: int, payload: bytes = b"") -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    magic = command ^ 0xFFFFFFFF
    header = struct.pack("<6I", command, arg0, arg1, len(payload), crc, magic)
    return header + payload


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return buf


def read_message(sock: socket.socket):
    header = recv_exact(sock, 24)
    command, arg0, arg1, length, _crc, _magic = struct.unpack("<6I", header)
    payload = recv_exact(sock, length) if length else b""
    return command, arg0, arg1, payload


def run_shell(host: str, port: int, command: str, timeout: float = 30.0) -> str:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    try:
        # ---- CNXN handshake ------------------------------------------------
        sock.sendall(pack(CNXN, VERSION, MAX_PAYLOAD,
                          b"host::features=shell_v2\x00"))
        command, arg0, arg1, payload = read_message(sock)
        if command != CNXN:
            raise RuntimeError("expected CNXN, got 0x%08x %r" % (command, payload))
        device_banner = payload.rstrip(b"\x00").decode("utf-8", "replace")
        max_data = arg1 or MAX_PAYLOAD

        # ---- OPEN the shell -------------------------------------------------
        sock.sendall(pack(OPEN, 7, 0, ("shell:" + command + "\x00").encode()))
        out = []
        closed = False
        local_id = 7
        while not closed:
            command, arg0, arg1, payload = read_message(sock)
            if command == OKAY:
                if arg0 != local_id:
                    raise RuntimeError("OKAY for the wrong channel: %d" % arg0)
                continue
            if command == WRTE:
                out.append(payload)
                sock.sendall(pack(OKAY, local_id, arg1, b""))
                continue
            if command == CLSE:
                closed = True
                continue
            if command == CNXN:
                continue
            # anything else is ignored
        return device_banner + "\n" + b"".join(out).decode("utf-8", "replace")
    finally:
        try:
            sock.close()
        except Exception:
            pass


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    host = sys.argv[1]
    port = int(sys.argv[2])
    command = " ".join(sys.argv[3:])
    try:
        print(run_shell(host, port, command))
    except Exception as exc:
        print("[adb_shell] FAILED: %s: %s" % (type(exc).__name__, exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
