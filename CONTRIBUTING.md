# Working on mustanad

This is a portfolio project, so it is not looking for feature contributions. This file exists for
a different reason: the README claims another developer could pick this code up, and that claim
should be checkable. If you are evaluating the project, reviewing it, or forking it for your own
use, everything you need is below.

## Setting up

Python 3.11 or newer. `.python-version` pins 3.13, which is what the tests were measured on.

```bash
make setup      # create .venv and install the package with dev extras
make check      # run every gate the CI runs
make demo       # index the sample corpus and ask questions in English and Arabic
```

`make help` lists every target. **Nothing in `make setup`, `make check` or `make demo` touches the
network.** The default provider is extractive and needs no API key, no model download and no
connection; a remote or local model is opt-in through configuration.

## The gates

`make check` runs the same four things CI runs, and all four must be clean:

| Gate | Command | Standard |
|---|---|---|
| Format | `ruff format --check .` | 100-column lines, no exceptions |
| Lint | `ruff check .` | the rule set in `pyproject.toml`, zero findings |
| Types | `mypy` | `disallow_untyped_defs`, so every function is annotated |
| Tests | `pytest` | 238 passing, zero skipped |

CI additionally runs `scripts/check_repo_hygiene.py`, which fails the build if a database, a
virtual environment or a credential-shaped literal has been committed. Run it locally with
`python scripts/check_repo_hygiene.py` before any commit.

Ruff's rule selection and the few deliberate `ignore` entries are documented inline in
`pyproject.toml`. `RUF001`–`RUF003` are off on purpose: they flag Arabic letters as "ambiguous"
Unicode, which in a bilingual project is nothing but false positives. Please do not add a bare
`# noqa` — if a rule genuinely does not apply, say why in a comment next to it.

## The parts that need care

**Arabic text handling (`text/normalize.py`, `text/stem.py`).** Two traps are already documented in
tests, and both were real bugs:

- A tashkeel character range written as a literal span can silently swallow Arabic-Indic digits.
  The ranges here are spelled out with explicit escapes for that reason — do not "simplify" them.
- The light stemmer must be **idempotent**. Single-pass suffix stripping turned `مجانيا` into
  `مجاني` and then `مجاني` into `مجان`, which breaks matching. It now iterates to a fixed point, and
  a test asserts that stemming twice equals stemming once.

**Retrieval quality.** The README reports measured numbers on a bilingual sample corpus. If you
change ranking, scoring or tokenisation, re-run the evaluation and update those numbers — leaving a
stale figure in the README is worse than reporting a lower one.

**Loaders (`loaders.py`).** These read attacker-controlled bytes. The DOCX guard inspects the ZIP
central directory *before* expanding anything, and refuses an archive declaring more than 64 MB
uncompressed or a ratio above 200:1. Any change here needs a test proving a crafted archive is
still refused.

## Tests

A behaviour change needs a test that fails before it and passes after. The suite is offline and
deterministic; if you need a clock or a model call, inject it rather than patching globals.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), as in the existing history:
`fix(loaders): …`, `feat(api): …`, `docs: …`. The body should explain *why*, since the diff already
shows *what*.

Do not commit a `.env`, an index database, or anything under `data/`. See
[SECURITY.md](SECURITY.md) for how to report a vulnerability.
