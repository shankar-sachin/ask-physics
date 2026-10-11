# Security Policy

## Supported versions

Ask Physics is pre-alpha. Only the latest release on `main` receives fixes.

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

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

This project feeds user text to our own Fermi language models, loads model
weights from disk, and executes symbolic math on the result. It holds no API
keys. The CLI makes no network calls except model downloads: the one at install
time, the one the first question runs if tellus or solem is missing, and celeste's,
which happens only after you say yes to it (or have saved "always"). Each downloaded file must
match the size and sha256 pinned in `src/askphysics/lm/weights.json`, or it is refused. The website
runs in the browser and loads Pyodide, KaTeX and fonts from CDNs. The scripts in
`scripts/` that build data and site assets fetch their sources over the network;
they are for contributors, not the CLI. The interesting attack surface is:

- **Prompt injection** that makes the pipeline ignore the retrieved context
  or emit fabricated equations presented as sourced. Constrained decoding
  limits plans to retrieved equation ids and numbers from the input, so a
  successful injection should be impossible by construction; any bypass is a
  vulnerability.
- **Unsafe weight loading.** Weights must be `safetensors`. Any code path
  that loads a pickled checkpoint (`torch.load`, `pickle.load`) is a
  vulnerability, because unpickling runs arbitrary code.
- **Unsafe expression parsing.** Equation and plan payloads are parsed with
  SymPy. Any path where untrusted text reaches `sympy.sympify` (which can call
  `eval`) without the restricted parser is a vulnerability. Seed data is
  parsed with `sympy.parsing.sympy_parser.parse_expr` and a whitelisted
  namespace; model output must never be `eval`'d.
- **Data supply chain.** A contributed equation, dataset, or set of weights
  carrying malicious payloads (for example, an expression string crafted to
  exploit the parser).

Wrong physics answers are bugs, not vulnerabilities. Open a regular issue for
those, and include the question you asked.

## Hygiene for contributors

- Ask Physics needs no API keys. If a change adds a network call or a key,
  it needs an ADR first (ADR-009 rules out external LLM services).
- Tests must run without trained weights, using `FakeLLMClient` or
  `fermi-luna-1`.
- Only load weights you trained or that come from this repo's release assets,
  with checksums verified.
