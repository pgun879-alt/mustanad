# Security policy

## Status of this project

`mustanad` is an open-source portfolio project. It has **no production deployment, no users and
no clients**, and it is maintained by one person in his spare time. Only the `main` branch is
supported; there are no releases and no backports.

That context sets your expectations, not the handling: this project parses untrusted files
uploaded by whoever can reach the API, so a flaw in the loaders or the auth layer is worth
reporting, and will be fixed and described honestly in the README rather than quietly patched.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** on this repository: the **Security** tab →
*Report a vulnerability*. That keeps the report private until a fix exists, and needs no email
address from either of us.

For anything that is not sensitive — a hardening suggestion, a question about the threat model, a
documentation error — open a normal issue instead.

Please include the version (commit SHA), the configuration involved, and the smallest input that
shows the problem. A failing test is the most useful form a report can take. **If your report
involves a malicious file, describe how to generate it rather than attaching it.**

**Expect best-effort, unpaid handling.** There is no SLA and no bug bounty. I will acknowledge a
report when I see it, and tell you plainly if I do not intend to fix something.

## In scope

The security model is described in the README's *Security* section. The most valuable reports
concern the file-parsing path, because that is the part that handles attacker-controlled bytes:

- **Resource exhaustion that evades the documented caps** — the DOCX guard refuses an archive
  declaring more than 64 MB uncompressed or a compression ratio above 200:1, PDFs are capped at
  2,000 pages, and uploads are capped by `max_upload_bytes`. A file that gets past all of these
  and still exhausts memory or CPU is a real finding.
- **Path traversal or arbitrary file write** through an upload filename or an archive member name.
- **Code execution through a parser** — anything that turns a crafted PDF or DOCX into execution
  rather than a refusal.
- **Authentication bypass** — reaching a protected endpoint without a valid API key when
  `auth_required` is on, or defeating the rate limiter.
- **Cross-tenant or cross-document exposure** — returning a passage from a document the caller
  never uploaded, where the deployment separates them.
- **XSS in the demo page** served at `/`, or in any answer or citation rendered into it.
- **Secret disclosure** — an API key appearing in a log line, an error response or an answer.
- **SSRF through a configured provider** — `llm_base_url` and `ollama_base_url` are operator
  settings, but a path that lets a *request* redirect them elsewhere is in scope.

## Deliberate decisions that are not vulnerabilities

Please read these before reporting, so neither of us wastes an afternoon:

- **Wrong or incomplete answers are not vulnerabilities.** This is retrieval plus optional
  generation, not an oracle. The design response to low confidence is to say the documents do not
  contain the answer, and the README states the retrieval quality that was actually measured.
- **Refusing to answer is the intended behaviour**, not a denial of service.
- **No network egress happens by default.** The extractive provider needs no API key and no
  connection. Configuring a remote provider is an explicit operator decision with a daily call
  budget attached.
- **Everything in `.env.example` is a placeholder.** A credential-shaped string there is not a
  leaked credential.
- **API keys are compared as SHA-256 digests held in memory.** They come from the environment by
  design; there is no key store to break into.
- **The sample corpus in `samples/` is synthetic.** It describes no real organisation, employee or
  policy.

## What is not covered

Issues in Python, FastAPI, `pypdf`, `python-docx` or any model provider themselves — report those
upstream. Vulnerabilities that require an attacker to already control the server, the index
database or the environment variables are out of scope, since every secret this project has lives
there.
