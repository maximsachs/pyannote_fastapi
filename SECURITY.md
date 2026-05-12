# Security Policy

## Supported versions

Only the latest `main` build and the most recent published image tag receive
security fixes.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Use GitHub's private vulnerability reporting on this repository:
**Security → Report a vulnerability** (the "Report a vulnerability" button on
the Security tab). This opens a private advisory visible only to the
maintainer.

Please include:

- A description of the issue and its impact.
- Steps to reproduce (proof-of-concept, request payload, image tag, etc.).
- Any suggested mitigation, if known.

You can expect an initial acknowledgement within a few business days. Once a
fix is available, a new image tag will be published and a GitHub Security
Advisory released.

## Scope

In scope:

- The FastAPI wrapper code in this repository (`app/`).
- The published Docker images built from this repository's `Dockerfile`s.
- The GitHub Actions workflows under `.github/workflows/`.

Out of scope:

- Vulnerabilities in upstream dependencies (`pyannote.audio`, `torch`,
  `fastapi`, base images, etc.) — please report those to the respective
  upstream projects. We will pick up patched releases via the normal rebuild
  flow.
- The pyannote model weights themselves and Hugging Face Hub infrastructure.

## Handling of secrets

This project never bakes secrets or model weights into published images.
`HF_TOKEN` and `API_KEYS` must be supplied at runtime by the deployer.
