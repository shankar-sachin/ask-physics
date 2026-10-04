# Security Policy

## Supported versions

Ask Physics is pre-alpha. Only the latest release on `main` receives fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Use GitHub's private vulnerability reporting instead:
**Security tab -> "Report a vulnerability"** on
<https://github.com/shankar-sachin/ask-physics/security/advisories/new>.

Please include:

- What you found and where (file, function, or command).
- Steps to reproduce, ideally a minimal question or input that triggers it.
- What you think the impact is.

You should get an acknowledgement within 7 days. Fixes for confirmed issues
ship in the next patch release, and reporters are credited in `CHANGELOG.md`
unless they ask not to be.

## What counts as a security issue here

This project sends user text to an LLM and executes symbolic math on the
result, so the interesting attack surface is:

- **Prompt injection** that makes the pipeline leak secrets, ignore the
  retrieved context, or emit fabricated equations presented as sourced.
- **Unsafe expression parsing.** Equation and plan payloads are parsed with
  SymPy. Any path where untrusted text reaches `sympy.sympify` (which can call
  `eval`) without the restricted parser is a vulnerability. Seed data is
  parsed with `sympy.parsing.sympy_parser.parse_expr` and a whitelisted
  namespace; LLM output must never be `eval`'d.
- **Secret leakage.** API keys belong in environment variables or a local
  `.env` file, which is git-ignored. Keys must never be logged, cached, or
  written into pipeline traces.
- **Data supply chain.** A contributed equation or dataset carrying malicious
  payloads (for example, an expression string crafted to exploit the parser).

Wrong physics answers are bugs, not vulnerabilities. Open a regular issue for
those, and include the question you asked.

## API key hygiene for contributors

- Copy `.env.example` to `.env`. Never commit `.env`.
- Tests must run with no API key, using `FakeLLMClient`. A test that needs a
  real key is a broken test.
- If you leak a key in a commit, rotate it immediately. Rewriting history is
  not enough; assume it has already been scraped.
