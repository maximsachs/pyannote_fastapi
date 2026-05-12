#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:?Set API_KEY (one of the keys from API_KEYS)}"
AUDIO="${AUDIO:-/path/to/audio.wav}"

curl -fsS "${BASE_URL}/live"

curl -fsS "${BASE_URL}/health"

curl -fsS "${BASE_URL}/metrics" | head

curl -fsS -N \
  -X POST "${BASE_URL}/diarize?num_speakers=2" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: text/event-stream" \
  -F "file=@${AUDIO};type=audio/wav"
