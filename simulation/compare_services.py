"""Compare two generators using isolated RPC sessions, never physical playback."""
import argparse
import json
from pathlib import Path
import urllib.request
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reference-url", required=True)
parser.add_argument("--candidate-url", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(url, path, body=None):
    request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=90) as response: return json.load(response)


health = [call(url, "/health") for url in [args.reference_url, args.candidate_url]]
for key in ["checkpoint_sha256", "stats_sha256", "control_contract", "padding_contract", "diffusion_steps", "cfg"]:
    assert health[0][key] == health[1][key], key
rows = []
for seed in (202610161, 202610162):
    sessions = []
    try:
        for url in [args.reference_url, args.candidate_url]:
            sessions.append((url, call(url, "/session", {"seed": seed})["session"]))
        for window in range(8):
            prompt = "A person continuously walks straight ahead with a natural gait." if window < 6 else "A person walks backward naturally."
            control = {} if window < 4 else {"root2d": [{"frame": 180, "xz": [.8, 1.5]}]}
            results = [call(url, "/next", {"session": sid, "window": window, "revision": 0,
                "prompt": prompt, "control": control, "request_version": window, "include_motion": True}) for url, sid in sessions]
            errors = {key: float(np.max(np.abs(np.array(results[0][key]) - np.array(results[1][key])))) for key in ["normalized_motion", "qpos"]}
            assert all(value == 0 for value in errors.values()), (seed, window, errors)
            rows.append(dict(seed=seed, window=window, max_abs=errors))
            for url, sid in sessions: call(url, "/publish", {"session": sid, "boundary": window + 1, "revision": 0})
    finally:
        for url, sid in sessions: call(url, "/close", {"session": sid})
record = dict(passed=True, rows=rows, health=health,
    scope="Same checkpoint/seed/text/controls; 8 windows x 2 seeds, crosses FIFO and changes text; isolated sessions not sent to physics")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2) + "\n")
print("SERVICE_PORTABILITY_PARITY_PASS", len(rows), "windows; exact motion and qpos")
