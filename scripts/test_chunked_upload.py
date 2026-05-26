#!/usr/bin/env python3
"""Integration script: chunked upload session → chunks → complete → SSE result."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import httpx


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, help="Mono WAV file to diarize")
    parser.add_argument("--base-url", default=os.environ.get("PYANNOTE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    if not args.api_key:
        print("Set --api-key or API_KEY", file=sys.stderr)
        return 2

    wav_bytes = args.wav_path.read_bytes()
    headers = {"Authorization": f"Bearer {args.api_key}"}

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=httpx.Timeout(3600.0)) as client:
        caps = client.get("/diarize/capabilities", headers=headers)
        caps.raise_for_status()
        print("capabilities:", caps.json())

        session = client.post(
            "/diarize/sessions",
            headers=headers,
            json={
                "filename": args.wav_path.name,
                "content_type": "audio/wav",
                "total_size_bytes": len(wav_bytes),
                "chunk_size_bytes": args.chunk_size,
            },
        )
        session.raise_for_status()
        meta = session.json()
        print("session:", meta)

        chunk_size = meta["chunk_size_bytes"]
        count = meta["expected_chunk_count"]
        upload_id = meta["upload_id"]
        for index in range(count):
            start = index * chunk_size
            end = min(start + chunk_size, len(wav_bytes))
            chunk = wav_bytes[start:end]
            put = client.put(
                f"/diarize/sessions/{upload_id}/chunks/{index}",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=chunk,
            )
            put.raise_for_status()
            print(f"chunk {index + 1}/{count} uploaded ({len(chunk)} bytes)")

        with client.stream(
            "POST",
            f"/diarize/sessions/{upload_id}/complete",
            headers={**headers, "Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            body = response.read().decode()
        events = _parse_sse(body)
        print("sse events:", [name for name, _ in events])
        if not any(name == "result" for name, _ in events):
            print(body, file=sys.stderr)
            return 1
        result = next(data for name, data in events if name == "result")
        print("result:", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
