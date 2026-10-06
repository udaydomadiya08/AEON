# Contributing to AEON

Thank you for your interest in contributing to AEON! We welcome contributions, bug reports, feature enhancements, and benchmark improvements from the community.

---

## 1. Code of Conduct

All contributors and maintainers are expected to adhere to the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## 2. Scientific Integrity & Provenance Policy

AEON is developed under strict standards of empirical honesty and reproducibility. All contributors must follow these non-negotiable research rules:

> **Core Research Rule**: Benchmark results and performance claims must preserve provenance and must not present simulated or estimated measurements as physical measurements.

1. **Explicit Provenance**: Any new metric added to reports, the dashboard, or telemetry must specify its provenance (`MEASURED`, `SIMULATED`, `HYBRID`, `ESTIMATED`, or `DERIVED`).
2. **No Overclaiming**: PRs must not claim universal cost reductions, latency speedups, or superior answer quality without rigorous experimental evidence.
3. **No Hardcoded Shortcuts**: Evaluation functions and controllers must never branch conditionally on test names or embed hardcoded benchmark targets. All code must pass the AST integrity scanner (`aeon benchmark validate`).

---

## 3. Development Setup

### Prerequisites
* Python 3.9 or higher
* Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/aeon.git
cd aeon

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 4. Running Tests

Before submitting any changes, verify that the complete test suite passes:

```bash
# Run all tests
pytest -q

# Run specific test modules
pytest tests/test_phase10.py -v
pytest tests/test_phase9.py -v

# Run the benchmark integrity scanner
python -m pytest tests/test_phase9.py -k "validate or reproducibility"
```

No pull request will be accepted if it weakens or removes existing regression tests.

---

## 5. Coding Standards

* **Code Style**: Follow standard PEP 8 conventions. Use clear, typed function signatures and comprehensive docstrings.
* **No Unbounded Memory**: Any cache, telemetry store, or queue must enforce explicit maximum capacity limits and age-based retention pruning.
* **Security & Secrets**: Never commit real API keys, credentials, tokens, or local filesystem paths. Use mock values and redact sensitive patterns.
* **Compatibility**: Preserve Python 3.9 compatibility. Do not introduce mandatory heavy dependencies without consensus.

---

## 6. Pull Request Process

1. **Fork and Branch**: Create a feature branch from `main` (`feature/my-enhancement` or `fix/issue-description`).
2. **Commit Messages**: Use clean, descriptive commit messages adhering to Conventional Commits:
   - `feat(controller): add adaptive temperature tuning`
   - `fix(server): handle client disconnect during streaming`
   - `docs(api): clarify extra_body parameters`
3. **Add Tests**: Every bug fix and new capability must include corresponding automated tests in `tests/`.
4. **Documentation**: Update relevant documentation files in `docs/` and `README.md` if public APIs or behaviors change.
5. **Open Pull Request**: Submit your pull request with a concise summary of the motivation, changes, and test results.

---

## 7. Reporting Issues

* **Bug Reports**: Open an issue detailing the exact environment (OS, Python version, hardware), reproduction steps, expected behavior, and observed behavior.
* **Security Vulnerabilities**: Do not disclose security vulnerabilities publicly on GitHub issues. Follow the instructions in [SECURITY.md](SECURITY.md).
