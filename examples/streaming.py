#!/usr/bin/env python3
"""
AEON Example: Server-Sent Events (SSE) Streaming.

Demonstrates streaming response chunks via `stream=True` against a local AEON server.
Reads `text/event-stream` chunks until the terminal `data: [DONE]` event.
"""

import json
import urllib.request


def run_streaming(base_url: str = "http://127.0.0.1:8000/v1"):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "aeon-auto",
        "messages": [
            {"role": "user", "content": "Explain Amdahl's Law in 3 bullet points."}
        ],
        "stream": True,
        "max_tokens": 128
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print(f"Connecting to stream at {url}...\n")
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                print("\n\n[Stream finished: data: [DONE]]")
                break
            if line.startswith("data: "):
                chunk_json = line[6:]
                try:
                    chunk = json.loads(chunk_json)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        print(delta["content"], end="", flush=True)
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        run_streaming()
    except Exception as e:
        print(f"Error streaming from AEON server: {e}")
        print("Ensure the server is running with: aeon serve")
