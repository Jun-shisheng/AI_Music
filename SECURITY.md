# Security policy

## Supported versions

Security fixes land on `main` and in the latest `v0.x.y` release. Older
pre-release snapshots are not maintained.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private
[Security Advisories](https://github.com/Jun-shisheng/AI_Music/security/advisories/new)
and include:

- the affected endpoint or file,
- a minimal reproduction (request, payload, or command),
- the impact you believe it has,
- the commit you tested.

You will get an acknowledgement, a severity assessment, and a fix or a
mitigation plan. Credit is given in the advisory unless you prefer otherwise.

## Scope

This project is a **single-user, local-only** tool. Both services bind to
`127.0.0.1`, there is no authentication, and the API is not designed to be
exposed to a network. Reports that assume the API is reachable from another
machine are out of scope unless they also work from localhost.

In scope:

- path traversal or arbitrary file read/write (for example through `/audio`),
- command injection or unsafe subprocess construction,
- SQL injection or database file escape,
- dependency vulnerabilities that are reachable from the default configuration,
- secrets or personal paths accidentally committed.

## Hardening notes

- `GET /audio/{filename}` serves only bare `.wav` names that resolve to a direct
  child of `OUTPUT_DIR`.
- The API runs a single worker process; the queue claims jobs atomically and
  recovers interrupted state on startup.
- Do not put real API keys in `.env.example`, issues, or logs.
- If you ever expose the studio beyond localhost, put it behind a reverse proxy
  with authentication and TLS; the application does not provide either.
