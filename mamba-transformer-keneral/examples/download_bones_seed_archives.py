"""Reliably download BONES-SEED archives over an unstable HTTP proxy."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


REPO_URL = "https://huggingface.co/datasets/bones-studio/seed/resolve/main"
OUTPUT = Path("/data4/zts/datasets/bones_seed")
ARCHIVES = ("g1.tar.gz", "soma_proportional.tar.gz", "soma_uniform.tar.gz")
EXPECTED_SIZES = {
    "g1.tar.gz": 23_499_973_647,
    "soma_proportional.tar.gz": 45_468_083_856,
    "soma_uniform.tar.gz": 45_188_088_297,
}
CHUNK_SIZE = 8 * 1024 * 1024


def download(name: str, token: str) -> None:
    output = OUTPUT / name
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists():
        if output.stat().st_size == EXPECTED_SIZES[name]:
            print(f"already_complete name={name} bytes={output.stat().st_size}", flush=True)
            return
        raise RuntimeError(
            f"completed file has the wrong size: {output}={output.stat().st_size}, "
            f"expected={EXPECTED_SIZES[name]}"
        )

    attempts = 0
    while True:
        attempts += 1
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Bearer {token}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=httpx.Timeout(connect=60.0, read=180.0, write=60.0, pool=60.0),
            ) as client:
                with client.stream("GET", f"{REPO_URL}/{name}?download=true", headers=headers) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if offset and not append:
                        print(f"range_not_honored name={name}; restarting", flush=True)
                        offset = 0
                    mode = "ab" if append else "wb"
                    expected = response.headers.get("content-range") or response.headers.get("content-length")
                    print(
                        f"transfer_start name={name} attempt={attempts} offset={offset} expected={expected}",
                        flush=True,
                    )
                    next_report = offset + 256 * 1024 * 1024
                    with partial.open(mode) as handle:
                        for chunk in response.iter_bytes(CHUNK_SIZE):
                            handle.write(chunk)
                            offset += len(chunk)
                            if offset >= next_report:
                                print(f"progress name={name} bytes={offset}", flush=True)
                                next_report = offset + 256 * 1024 * 1024
            if partial.stat().st_size != EXPECTED_SIZES[name]:
                raise IOError(
                    f"incomplete response for {name}: got {partial.stat().st_size}, "
                    f"expected {EXPECTED_SIZES[name]}"
                )
            partial.replace(output)
            print(f"transfer_complete name={name} bytes={output.stat().st_size}", flush=True)
            return
        except Exception as exc:
            print(
                f"transfer_retry name={name} attempt={attempts} offset={offset} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(min(60, 5 * attempts))


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not available in the environment")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(ARCHIVES)) as executor:
        list(executor.map(lambda archive: download(archive, token), ARCHIVES))
    print("all_archives_complete", flush=True)


if __name__ == "__main__":
    main()
