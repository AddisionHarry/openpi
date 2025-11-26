#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import os
import argparse
import threading

def recv_discarder(sock):
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
    except Exception:
        pass

def run_client(server_ip, port, mode="latency", count=10, duration=10):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, port))
    print(f"Connected to server {server_ip}:{port} with mode={mode}")

    if mode in ["latency", "both"]:
        latencies = []
        for i in range(count):
            start = time.time()
            msg = f"ping {i}".encode()
            client.sendall(msg)
            response = client.recv(1024).decode(errors="ignore")
            end = time.time()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            print(f"[{i}] Sent: {msg.decode()}, Received: {response}, Latency: {latency_ms:.2f} ms")
            time.sleep(1)
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average latency: {avg_latency:.2f} ms")

    if mode in ["bandwidth", "both"]:
        discard_thread = threading.Thread(target=recv_discarder, args=(client,), daemon=True)
        discard_thread.start()

        payload = os.urandom(4096)
        total_bytes = 0
        start_time = time.time()
        while time.time() - start_time < duration:
            client.sendall(payload)
            total_bytes += len(payload)
        end_time = time.time()
        seconds = end_time - start_time
        mbps = (total_bytes * 8) / (seconds * 1e6)
        print(f"Sent {total_bytes/1e6:.2f} MB in {seconds:.2f} s = {mbps:.2f} Mbps")

    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--port", type=int, default=1590)
    parser.add_argument("--mode", choices=["latency", "bandwidth", "both"], default="latency")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()
    run_client(args.server_ip, args.port, args.mode, args.count, args.duration)
