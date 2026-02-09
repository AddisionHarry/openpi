#!/usr/bin/env python3
"""
net_test.py

TCP client/server network performance test tool, compatible with older servers.

Client-side displays tqdm progress bars and unified ASCII-only report.

Usage:

Server:
    python net_test.py server --port <listen_port>

Client:
    python net_test.py client --host <server_ip> --port <server_port>
"""

import argparse
import os
import socket
import statistics
import time
from typing import List, Dict

from tqdm import tqdm

LATENCY_ROUNDS = 200
BANDWIDTH_MIB = 512
CHUNK_SIZE = 256 * 1024
MIB = 1024 * 1024


# -------------------------
# Server
# -------------------------
def run_server(port: int) -> None:
    """Run TCP server for latency echo and bandwidth receive tests."""
    print(f"[SERVER] Listening on 0.0.0.0:{port}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.listen(1)

        conn, addr = s.accept()
        print(f"[SERVER] Connected from {addr}")

        with conn:
            while True:
                try:
                    header = conn.recv(8)
                    if not header:
                        break
                    mode = header.decode(errors="ignore")
                except Exception:
                    break

                if mode.startswith("LATENCY"):
                    payload = conn.recv(4)
                    conn.sendall(payload)
                elif mode.startswith("BANDWID"):
                    total_bytes = 0
                    start = time.time()
                    while True:
                        data = conn.recv(CHUNK_SIZE)
                        if not data:
                            break
                        total_bytes += len(data)
                    elapsed = time.time() - start
                    mib_total = total_bytes / MIB
                    bw = mib_total / elapsed if elapsed > 0 else 0.0
                    print(f"[SERVER] Bandwidth recv: {mib_total:.2f} MiB in {elapsed:.2f}s ({bw:.2f} MiB/s)")
                    break
                else:
                    print(f"[SERVER] Unknown mode {mode}")
                    break
    print("[SERVER] Exit")


# -------------------------
# Client
# -------------------------
def run_client(host: str, port: int) -> None:
    """Run TCP latency and bandwidth tests with tqdm progress bars."""
    print(f"[CLIENT] Connecting to {host}:{port}")
    rtts: List[float] = []
    bw_samples: List[float] = []
    report: Dict[str, float] = {}

    total_bytes_target = BANDWIDTH_MIB * MIB

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((host, port))

        # ---------- Latency ----------
        print("\n[Latency Test]")
        pbar = tqdm(total=LATENCY_ROUNDS, desc="Latency RTT", unit="round")
        for _ in range(LATENCY_ROUNDS):
            payload = b"ping"
            t0 = time.perf_counter()
            s.sendall(b"LATENCY")
            s.sendall(payload)
            s.recv(len(payload))
            t1 = time.perf_counter()
            rtt_ms = (t1 - t0) * 1000.0
            rtts.append(rtt_ms)
            pbar.set_postfix(rtt_ms=f"{rtt_ms:.2f}", avg_ms=f"{statistics.mean(rtts):.2f}")
            pbar.update(1)
        pbar.close()

        report["latency_avg_ms"] = statistics.mean(rtts)
        report["latency_min_ms"] = min(rtts)
        report["latency_max_ms"] = max(rtts)
        report["latency_var_ms2"] = statistics.pvariance(rtts)
        report["latency_std_ms"] = statistics.pstdev(rtts)

        # ---------- Bandwidth ----------
        print("\n[Bandwidth Test]")
        s.sendall(b"BANDWID")
        block = os.urandom(CHUNK_SIZE)
        sent_bytes = 0
        start = time.time()
        last_time = start
        last_bytes = 0

        pbar = tqdm(total=BANDWIDTH_MIB, desc="Bandwidth send", unit="MiB")
        while sent_bytes < total_bytes_target:
            send_size = min(CHUNK_SIZE, total_bytes_target - sent_bytes)
            s.sendall(block[:send_size])
            sent_bytes += send_size

            progress_mib = int(sent_bytes / MIB)
            if progress_mib > pbar.n:
                pbar.n = progress_mib
                now = time.time()
                avg_bw = progress_mib / (now - start) if now > start else 0.0
                pbar.set_postfix(avg_mib_s=f"{avg_bw:.2f}")
                pbar.refresh()

        pbar.n = BANDWIDTH_MIB
        pbar.refresh()
        pbar.close()
        s.shutdown(socket.SHUT_WR)

        elapsed = time.time() - start
        report["bandwidth_mib"] = sent_bytes / MIB
        report["bandwidth_time_s"] = elapsed
        report["bandwidth_avg_mib_s"] = report["bandwidth_mib"] / elapsed if elapsed > 0 else 0.0
        report["bandwidth_var_mib_s2"] = statistics.pvariance(bw_samples) if bw_samples else 0.0
        report["bandwidth_std_mib_s"] = statistics.pstdev(bw_samples) if bw_samples else 0.0

    # ---------- Report ----------
    print("\n==============================")
    print("Network Test Report")
    print("==============================")
    print(f"Latency avg      : {report['latency_avg_ms']:.2f} ms")
    print(f"Latency min      : {report['latency_min_ms']:.2f} ms")
    print(f"Latency max      : {report['latency_max_ms']:.2f} ms")
    print(f"Latency variance : {report['latency_var_ms2']:.4f} ms^2")
    print(f"Latency stddev   : {report['latency_std_ms']:.4f} ms")
    print("")
    print(f"Bandwidth total  : {report['bandwidth_mib']:.2f} MiB")
    print(f"Bandwidth time   : {report['bandwidth_time_s']:.2f} s")
    print(f"Bandwidth avg    : {report['bandwidth_avg_mib_s']:.2f} MiB/s")
    print("==============================\n")


# -------------------------
# Entry
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="role", required=True)

    ps = sub.add_parser("server")
    ps.add_argument("--port", type=int, required=True)

    pc = sub.add_parser("client")
    pc.add_argument("--host", type=str, required=True)
    pc.add_argument("--port", type=int, required=True)

    args = parser.parse_args()
    if args.role == "server":
        run_server(args.port)
    else:
        run_client(args.host, args.port)


if __name__ == "__main__":
    main()
