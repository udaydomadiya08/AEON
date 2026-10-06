#!/usr/bin/env python3
"""
AEON Example: Basic Chat Completion.

Demonstrates a standard OpenAI-compatible Chat Completion request to a local AEON server.
Can be executed with either the official `openai` Python SDK or Python's standard `urllib`.
"""

import json
import urllib.request


def run_with_urllib(base_url: str = "http://127.0.0.1:8000/v1"):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "aeon-auto",
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": "What is 17 * 23?"}
        ],
        "temperature": 0.0,
        "max_tokens": 128
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print(f"Sending request to AEON server at {url}...")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("\n--- Model Response ---")
        print(data["choices"][0]["message"]["content"])
        print("\n--- AEON Orchestration Telemetry (x_aeon) ---")
        if "x_aeon" in data:
            for k, v in data["x_aeon"].items():
                print(f"  {k:<25}: {v}")


if __name__ == "__main__":
    try:
        run_with_urllib()
    except Exception as e:
        print(f"Error querying AEON server: {e}")
        print("Ensure the server is running with: aeon serve")
