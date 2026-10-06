# Security Policy

## 1. Supported Versions

Security updates and patches are provided for the current active release:

| Version | Supported |
| :--- | :--- |
| `0.1.x` | :white_check_mark: |
| `< 0.1.0` | :x: |

---

## 2. Reporting a Vulnerability

We take the security of AEON seriously. If you discover a security vulnerability, please disclose it responsibly:

* **Do NOT report security vulnerabilities via public GitHub issues, discussions, or pull requests.**
* Please report vulnerabilities via [GitHub Private Vulnerability Reporting](https://github.com/your-org/aeon/security/advisories/new) on the repository.
* Include a detailed description of the vulnerability, reproduction steps, proof-of-concept payload, and the potential impact.
* The maintainers will acknowledge receipt within 48 hours and work with you to validate and remediate the issue prior to public disclosure.

---

## 3. Local-First Security Architecture

AEON is designed with defensive default settings:

* **Localhost Default**: The AEON server binds strictly to `127.0.0.1` by default. Binding to `0.0.0.0` or a public network interface displays an explicit security warning.
* **Authentication**: When `AEON_API_KEY` is configured, incoming requests must supply a valid key via the `Authorization: Bearer <key>` or `X-API-Key: <key>` header. Verification uses `hmac.compare_digest` to prevent timing attacks.
* **Secret Scrubbing**: Telemetry logging scrubs common secret patterns (API keys, authorization tokens, passwords, session cookies) before persisting events or displaying metrics in the dashboard.
* **Sandboxed Tools**: Deterministic tools (Calculator, Python sandbox, SymPy) operate within restricted execution environments to prevent arbitrary shell command injection or unauthorized filesystem operations.
* **Bounded Resources**: In-memory rate limiting, request payload bounds (`1MB` default), and backpressure queues prevent resource exhaustion and Denial-of-Service attacks.
