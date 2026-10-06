#!/usr/bin/env python3
"""
AEON Example: Constrained Request with Policy Overrides.

Demonstrates supplying namespaced AEON policy controls via `extra_body.aeon`
to enforce strict quality targets, latency SLAs, cost budgets, and preferred hardware.
"""

import json
import urllib.request


def run_constrained(base_url: str = "http://127.0.0.1:8000/v1"):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "aeon-auto",
        "messages": [
            {"role": "user", "content": "Compute matrix multiplication properties concisely."}
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "extra_body": {
            "aeon": {
                "quality_target": 0.95,
                "latency_sla_ms": 350.0,
                "cost_budget_usd": 0.0001,
                "preferred_device": "cpu",
                "preferred_precision": "fp32"
            }
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print(f"Sending constrained request to {url}...")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("\n--- Model Response ---")
        print(data["choices"][0]["message"]["content"])
        print("\n--- AEON Execution & Policy Decisions ---")
        meta = data.get("x_aeon", {})
        print(f"  Selected Model       : {meta.get('selected_model')}")
        print(f"  Selected Device      : {meta.get('selected_device')}")
        print(f"  Selected Precision   : {meta.get('selected_precision')}")
        print(f"  Executed Layer Depth : {meta.get('selected_depth')}")
        print(f"  Latency              : {meta.get('latency_ms', 0):.2f} ms")
        print(f"  Estimated Cost       : ${meta.get('estimated_cost_usd', 0):.7f}")
        print(f"  Provenance           : {meta.get('provenance')}")


if __name__ == "__main__":
    try:
        run_constrained()
    except Exception as e:
        print(f"Error querying AEON server: {e}")
        print("Ensure the server is running with: aeon serve")
