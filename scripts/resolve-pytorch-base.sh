#!/usr/bin/env bash
# Resolve a pytorch/pytorch base image tag that satisfies pyannote.audio's
# declared torch requirement.
#
# Usage:
#   PYANNOTE_VERSION=4.0.4 ./scripts/resolve-pytorch-base.sh
#   (or omit PYANNOTE_VERSION to use the latest from PyPI)
#
# Output (stdout): a single fully-qualified base image reference, e.g.
#   pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
#
# Strategy:
#   1. Pull pyannote.audio metadata from PyPI; extract the torch lower bound
#      from requires_dist (e.g. "torch>=2.8.0" -> "2.8.0").
#   2. List pytorch/pytorch tags matching '<X.Y.Z>-cuda<C>-cudnn9-runtime'.
#   3. Keep only tags whose torch version >= the required lower bound.
#   4. Pick the torch version per TORCH_SELECT:
#        - "min" (default): lowest torch >= floor. Matches what pyannote was
#          tested against, minimizes NVIDIA driver requirement for end users.
#        - "max": highest available. Newest features/perf, may require newer
#          NVIDIA drivers.
#      Within the chosen torch version, prefer CUDA_PREFERRED (default 12.8),
#      else fall back to the highest CUDA available.
set -euo pipefail

CUDA_PREFERRED="${CUDA_PREFERRED:-12.8}"
TORCH_SELECT="${TORCH_SELECT:-min}"
PYANNOTE_VERSION="${PYANNOTE_VERSION:-}"

if [ -z "${PYANNOTE_VERSION}" ]; then
  PYANNOTE_VERSION="$(curl -fsSL https://pypi.org/pypi/pyannote.audio/json | jq -r '.info.version')"
fi

PYANNOTE_JSON="$(curl -fsSL "https://pypi.org/pypi/pyannote.audio/${PYANNOTE_VERSION}/json")"

TORCH_MIN="$(
  echo "${PYANNOTE_JSON}" \
    | jq -r '.info.requires_dist[]? | select(test("^torch[^a-zA-Z_]")) | capture("torch[ ]*>=[ ]*(?<v>[0-9]+(\\.[0-9]+){1,2})") | .v' \
    | head -n1
)"

if [ -z "${TORCH_MIN}" ]; then
  echo "Could not parse torch>= bound from pyannote.audio==${PYANNOTE_VERSION}" >&2
  exit 1
fi

vercmp() {
  printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1
}
vge() {
  [ "$(vercmp "$1" "$2")" = "$2" ]
}

TAGS_JSON="$(curl -fsSL "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags?page_size=100&ordering=last_updated")"

CANDIDATES="$(
  echo "${TAGS_JSON}" \
    | jq -r '.results[].name' \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+-cuda[0-9.]+-cudnn9-runtime$' \
    || true
)"

if [ -z "${CANDIDATES}" ]; then
  echo "No matching pytorch/pytorch tags found" >&2
  exit 1
fi

TORCH_VERSIONS="$(
  echo "${CANDIDATES}" \
    | awk -F'-' '{print $1}' \
    | sort -V -u
)"

ELIGIBLE=""
while IFS= read -r v; do
  if vge "${v}" "${TORCH_MIN}"; then
    ELIGIBLE="${ELIGIBLE}${v}"$'\n'
  fi
done <<< "${TORCH_VERSIONS}"
ELIGIBLE="${ELIGIBLE%$'\n'}"

if [ -z "${ELIGIBLE}" ]; then
  echo "No pytorch/pytorch tag satisfies torch>=${TORCH_MIN}" >&2
  exit 1
fi

case "${TORCH_SELECT}" in
  min) BEST_TORCH="$(echo "${ELIGIBLE}" | head -n1)" ;;
  max) BEST_TORCH="$(echo "${ELIGIBLE}" | tail -n1)" ;;
  *)
    echo "Unknown TORCH_SELECT=${TORCH_SELECT} (use min or max)" >&2
    exit 1
    ;;
esac

BEST_TAG=""
while IFS= read -r tag; do
  torch="${tag%%-*}"
  cuda="$(echo "${tag}" | sed -E 's/^[0-9.]+-cuda([0-9.]+)-cudnn9-runtime$/\1/')"
  [ "${torch}" = "${BEST_TORCH}" ] || continue
  if [ "${cuda}" = "${CUDA_PREFERRED}" ]; then
    BEST_TAG="${tag}"
    break
  fi
  if [ -z "${BEST_TAG}" ]; then
    BEST_TAG="${tag}"
  fi
done <<< "${CANDIDATES}"

if [ -z "${BEST_TAG}" ]; then
  echo "No pytorch/pytorch tag satisfies torch>=${TORCH_MIN}" >&2
  exit 1
fi

echo "Resolved: pyannote.audio=${PYANNOTE_VERSION} requires torch>=${TORCH_MIN}" >&2
echo "Selected base image tag: ${BEST_TAG}" >&2
echo "pytorch/pytorch:${BEST_TAG}"
